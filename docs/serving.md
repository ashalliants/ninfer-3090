# HTTP serving

`build/apps/ninfer-serve` loads one v3 `.ninfer` artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype int8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
when running alone. Requests acquire KV pages as execution advances; if concurrent growth exhausts
the pool, the scheduler can pause a request and restore it later.

With `C=2` and two extra Device slots, the process owns four Device StateImages. The default shared
pinned Host budget is 8 GiB plus eight model StateImages. It holds retained state, KV and pause
snapshots, including in-flight destinations; `--host-context-mib` sets an explicit total instead.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Add
`--vision` when the server must accept image or video input. Speculative residency is likewise
frozen by `--spec mtp|dflash|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend.
Any `--spec` also loads the optimized proposal head (`--lm-head-draft` is implied). DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
with `--vision`; each accelerates generated-text decode after multimodal prefill, while Vision encode
and prefill remain outside speculative acceleration. A later request cannot enable a capability
omitted at startup. The artifact need only contain the Text backbone and the optional components
selected for this process.

### Options from a file

`--config FILE` reads serve options from a JSON object whose keys are the flag names without their
leading dashes, so a setup can be kept in a file and started the same way each time:

```json
{
  "max-context": 65536,
  "kv-capacity": 65536,
  "kv-dtype": "int8",
  "spec": "mtp",
  "draft-tokens": 3,
  "lm-head-draft": true,
  "preserve-thinking": true,
  "cors": false,
  "allowed-host": ["gpubox.lan"]
}
```

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer --config serve.json --max-context 32768
```

- A string or a number is the flag's value. `true` is a bare flag (`"lm-head-draft": true` is
  `--lm-head-draft`) and `false` leaves the option at its default. An array repeats the flag, and
  only the options that collect their occurrences take one (`graft`, `allowed-host`,
  `context-store-peer`).
- The command line wins over the file wherever `--config` stands, and the file over the defaults.
  A repeatable option given in both keeps the file's entries and adds the command line's.
- The file is the command line written down: a relative path in it is read against the working
  directory, as the same word on the command line would be, not against the file's own directory.
- A key no option has fails at startup with `FILE: unknown key "KEY"`; so do a value an option
  refuses (`FILE: "port": invalid port: eighty`), a value for a switch, `true` or `false` for an
  option that takes a value, a nested object and `null`.
- `api-key` is refused in the file: a secret belongs on the command line or in the launcher's
  environment, not in a file that is copied and shared. A file cannot name another (`config`), and
  one `--config` is accepted per command line.
- The launchers' profiles are such files: `scripts/profiles/` (`profiles/` in a release archive);
  [launcher profiles](maintainer/launcher-profiles.md) says what each holds.
- A switch the file turns on can be turned off from the command line only where its `no-` flag
  exists (`--no-models-autoload` against `"models-autoload": true`). Where none exists, the attempt
  fails rather than leaving the switch on: `--no-vision` against `"vision": true` stops startup with
  `FILE: "vision" is set and has no command-line override`. Edit the file, or use another one.
- In router mode the file's options are part of the command line passed on to every model, so they
  win over a preset's sections as the command line does, and the router's own options
  (`models-dir`, `models-max`, ...) may be given in it too.

At startup the server logs where each option came from, the defaults left out:

```text
INFO  settings | command line: --config, --max-context | serve.json: --allowed-host, --draft-tokens, --kv-capacity, --kv-dtype, --lm-head-draft, --preserve-thinking, --spec
```

The `server_start` record of `--request-log-jsonl` carries the same as `setting_sources` (each
option given: `cli`, the file's path, or `default` for a key the file set to `false`) beside
`effective_argv`, the words the options were parsed from.

### Qwen3.8-Flash-Next (preview)

The one command:

```bat
set NINFER_MODEL=K:\ninfer-models\qwen4exp\qwen4exp-iq2xs-mtp-vision.ninfer
scripts\run.bat qwen38-flash-next
```

That launcher profile runs `ninfer-serve <model> --max-concurrency 1 (2 with NINFER_CONCURRENCY=2) --max-context 32768
--kv-capacity 32768 --kv-dtype int8 --prefill-chunk 1024`, the only serving shape this release
supports, adding `--spec mtp --draft-tokens 3` when the model is `qwen4exp-iq2xs-mtp.ninfer` or
`qwen4exp-iq2xs-mtp-vision.ninfer` (the artifacts with the MTP drafter) and `--vision` when it is a
`-vision` artifact (converted with the vision tower, `--mmproj`). Without `NINFER_MODEL` the launcher
prefers `-mtp-vision`, then `-mtp`, when they sit beside the plain one; `NINFER_SPEC=none` turns
speculation off and `NINFER_VISION=off` image input. `NINFER_CONTEXT` changes the context (up to the
model's positions; KV above 32,768 tokens is borrowed from the expert cache). What is different from
the Qwen3.5 models:

- **One request at a time by default; two lanes opt-in** (`--max-concurrency 2`, `NINFER_CONCURRENCY=2` in the launcher). `--max-concurrency` above 2, `--kv-dtype` other than `int8`,
  `--spec dflash|dflash2`, `--ngram-draft-tokens`, `--draft-context-rows`, `--graft`, `--auto-host-cache` and the load-time
  weight flags are refused at startup with the flag to change.
- **Images** (`--vision`, an artifact converted with `--mmproj`; otherwise startup refuses with that
  instruction): OpenAI `image_url` parts (data URIs, local files) and Anthropic image blocks, through
  the same media handling and processor as the Qwen3.5 models. The vision tower (0.84 GiB) stays
  pinned in host RAM and costs no VRAM between images; for each prompt with images the expert cache
  lends about 130 MiB plus the images' workspace while they are encoded, and a few MiB for their
  embeddings until the prompt has been read. A 768 x 512 image is 384 tokens and encodes in about
  50-100 ms; a 1920 x 1200 one, 2,280 tokens, in about 0.4 s. Startup encodes one blank image (about
  2 s) so the first real image does not pay the tower's one-time setup. `--vision-residency` has no
  effect for this model.
- **MTP speculation** (`--spec mtp --draft-tokens N`, N from 1 to 7) needs an artifact converted
  with the drafter (`--mtp-gguf`, [weight-conversion.md](weight-conversion.md)); a plain artifact
  refuses with that instruction. The output is the same as without it, token for token (greedy).
  The drafter keeps its 512 experts in VRAM (about 0.9 GiB, taken from the expert cache). `N` is a
  ceiling: each round drafts as many tokens as the recent acceptance pays for, mostly two on prose;
  a ceiling of 3 measured best or tied against 2 and against deeper fixed drafts (the
  [plan's PR 13 measurements](maintainer/flash-next-plan.md#pr-13-measurements)), about 1.4x plain decode.
  [Relaxed drafting](#relaxed-drafting) (`--relaxed-drafting`) applies to its verification rounds
  as to the Qwen3.5 models' and changes the output distribution the same way.
- **Prefix reuse** through the same context cache as the other models, with checkpoints in pinned
  host RAM or on local disk ([design](maintainer/flash-next-context-cache.md)). A conversation's next turn
  (or an agent's next tool round) restores the previous turn's state and prefills only the new
  tokens; repeating a prompt exactly samples its first token from the stored logits. A checkpoint is
  about 111 MiB of recurrent state plus 0.89 MiB per 64 tokens of KV (a 32K-token one 567 MiB), and
  checkpoints of one conversation share their KV pages. The host tier is 4 GiB unless
  `--host-context-mib N` says otherwise; it is planned in the RAM ledger beside the experts, so on a
  machine with less RAM to spare it shrinks to what fits (one startup line says so) and is off below
  512 MiB. `--host-context-mib 0` or `--no-prefix-reuse` turn reuse off; in the launcher,
  `NINFER_PREFIX_CACHE=off` (and `NINFER_HOST_CONTEXT_MIB` for the size). Restores and captures are
  host-to-device copies of a few milliseconds; nothing is kept in VRAM, so the expert cache keeps
  every frame.
- **Checkpoints on disk** (`--context-nvme-mib N`): a scratch file of up to N MiB holds the
  checkpoints, in `--context-nvme-dir DIR` or beside the model (put it on NVMe). It is created at
  startup, grows only with what is written, is capped at half the drive's free space, and is
  deleted when the server exits (also when it is killed). A tier under 512 MiB, or on a drive whose
  free space cannot be read, is not opened and checkpoints stay in RAM. The host tier above becomes its RAM cache:
  recent checkpoints restore from RAM, older ones from the file. With `--host-context-mib 0` every
  checkpoint lives on disk (disk-only reuse, for a machine with no RAM to spare). A full or failing
  disk turns the affected checkpoints into cache misses, never into wrong output.
- **Across restarts** (`--context-store DIR`): conversations are written to the context store when
  idle and at shutdown and restored at startup (or when a request for them arrives), exactly as for
  the other models; it needs the host tier or the disk tier. Images are tied to the model file, the
  MTP setting and `--vision`; a session written under another configuration is not restored.
- **Host RAM.** All 33 GiB of routed experts are pinned in host RAM at startup (with `--vision` also
  the 0.84 GiB tower, and the ledger counts up to 1.4 GiB of decoded media: about 2.2 GiB more in
  all). Before anything is pinned, a RAM ledger checks that the experts, every other planned allocation and a reserve for
  Windows (`--ram-headroom-mib`, default 2048) fit in the memory available now; if not, startup
  refuses with the shortfall ("... 2.9 GiB short. Close other programs ..."). Loading takes about
  two minutes from a SATA SSD, less when the files are in the page cache. With
  `--partial-ram-residency` RAM holds only the experts the VRAM cache does not (see the table
  below), so a pack whose experts exceed RAM can still load.
- **The VRAM expert cache.** Whatever VRAM the dense weights, the KV and the workspace leave, less
  a headroom for the desktop, holds the most-used experts (about 11,000-12,500 of the 24,576). Its
  state is saved beside the model (`<model>.expert-state`) at shutdown and every ten minutes
  between requests, and loaded at the next start, so the second start is warm. A cold cache fills
  at one promotion per layer per decoded token; decode speeds up as it fills. A model release may also ship a
  default state (`<model>.expert-state.default`) for the first start: it is read only when the
  saved state is missing or belongs to another artifact, it is never written, and
  `--no-expert-state` skips it too. No such file ships yet; see [A default expert
  state](#a-default-expert-state). Each finished
  request logs one line with its expert hit rate, promotions and decode speed.
- **Prefill.** A prompt of 512 tokens or more prefills in calls of 4,096 tokens (independent of
  `--prefill-chunk`, which sizes shorter prompts and forced tokens). For the prompt, the expert cache
  lends up to about 2,250 of its frames (3.2 GiB): the calls' workspace, sized to the prompt's
  widest call (a shorter prompt lends less), and a ring into which each layer's non-resident experts
  are copied two layers ahead while the GPU computes. The cache promotes nothing while the frames are
  lent. When the prompt is done the frames come back and the experts they displaced are restored
  first (the set decode had before the prompt), then any remaining free frames are filled with the
  experts the prompt used most, so the first decode tokens find them resident. A cache too small to
  lend what the prompt needs prefills in
  `--prefill-chunk` calls instead (slower, same output). The next call's n-gram rows are read while
  the GPU runs the current one; a prompt's first call waits for its own rows, so the volume's disk
  speed shows in time to first token.
- **The n-gram volume** `<model>.ngram` (29 GB) is read unbuffered per token, with a 2^20-row host
  cache, so its drive sets prefill speed; the 40 GB artifact is read only once, at load. Put the
  volume on the fastest drive, even when the model cannot go there: `--ngram-volume FILE` (in the
  launcher, `NINFER_NGRAM_VOLUME`) reads a copy anywhere, and the server refuses a volume that does
  not belong to the model. The plain and `-mtp` artifacts converted with `--ngram-reuse` share one
  volume, so one copy serves both. Measured 2026-10-12 on this RTX 3090 with the plain artifact
  (no speculation) on a SATA SSD, moving only the volume to an NVMe drive (warm requests,
  n = 12 / 12 / 6 per drive; prefill medians and the mean of the per-prompt ratios; prefill with
  MTP on was not measured):

  | Prompt | Volume on SATA | Volume on NVMe | NVMe / SATA, per prompt |
  |---|---:|---:|---:|
  | 4K | 1,825 tok/s | 2,345 tok/s | 1.37x |
  | 32K | 2,332 tok/s | 2,422 tok/s | 1.11x |
  | 128K | 2,135 tok/s | 2,372 tok/s | 1.13x |

  Plain decode at 8K moved +0.8% (it reads few rows). The SATA arm also varies more from prompt to
  prompt (4K: 1,334-2,128 tok/s against 2,320-2,393), and its first request in a process pays
  0.86 s of reads where NVMe pays 0.07 s.

  ```bat
  set NINFER_MODEL_DIR=K:\ninfer-models\qwen4exp
  set NINFER_NGRAM_VOLUME=C:\ninfer-models\qwen4exp\qwen4exp-iq2xs.ninfer.ngram
  scripts\run.bat qwen38-flash-next
  ```

| Flash-Next option | Meaning | Default |
|---|---|---|
| `--ngram-volume FILE` | the per-layer n-gram embedding volume | `<model>.ngram` |
| `--ngram-keepalive-ms N` | read one block of the n-gram volume after N ms idle so the drive does not sleep (SSDs sleep after about 100 ms); `0` turns it off; changes timing only | `40` (was a fixed 100) |
| `--expert-state FILE` | where the expert cache's state is saved and read | `<model>.expert-state` |
| `--no-expert-state` | start the expert cache empty and save nothing | off |
| `--default-expert-state FILE` | a default expert state for a fresh install, read only when the saved state is missing or belongs to another artifact (it is matched by artifact id, so it fits every copy of the model) and never written; absent file: nothing changes; `--no-expert-state` does not read it either | `<model>.expert-state.default` |
| `--vram-headroom-mib N` | VRAM the expert cache leaves free for the desktop and other programs | 512 with a display, 256 without |
| `--ram-headroom-mib N` | host RAM left free for Windows once every expert is pinned; startup refuses when it cannot be kept | `2048` |
| `--partial-ram-residency` | keep in host RAM only the routed experts the VRAM expert cache does not hold, plus 16 spare slots per layer, instead of pinning all 24,576: at startup the hottest experts (by the saved expert state, else an equal share of every layer) are read from the model file straight into VRAM and live only there, and every other expert gets a pinned RAM slot. A promoted expert keeps its RAM copy; an expert that lives only in VRAM is copied back into a RAM slot before its VRAM frame is reused. The output is designed to stay bit for bit unchanged. The cost: the expert cache keeps its startup size (it no longer shrinks when the desktop needs VRAM, so the default `--vram-headroom-mib` with a display is 1 GiB instead of 512 MiB), the whole KV of `--max-context` is mapped at startup (fewer frames for short prompts), prompts prefill through static calls without the expert stream (long prompts are slower), and the cache admits experts one victim at a time even when a round could admit every miss (an expert held only in VRAM leaves only when its layer has a RAM slot for it). Designed to work with two lanes, MTP, `--hot-embedding-rows`, `--cpu-verify-misses` and `--prefill-cpu-share` (the CPU computes only experts that are not in VRAM, and each of those has a RAM copy). Refused with `--vision`. RAM needed for the experts, estimated from the packs' expert sizes with about 15 GiB of VRAM frames (not measured): the IQ2_XS pack about 20 GiB (33 GiB whole), the IQ3_XXS pack about 30 GiB (40 GiB), the IQ3_S pack about 37 GiB (46.8 GiB); the context cache, the n-gram row cache and `--ram-headroom-mib` come on top. So the IQ3_XXS pack and probably the IQ3_S pack fit a 64 GB Linux machine, and the IQ2_XS pack a 32 GB one with the context cache shrunk. On Windows, when everything pinned exceeds half the RAM less 8 GiB (24 GiB on 64 GB, 8 GiB on 32 GB), startup prints a warning naming the `--vram-headroom-mib` or `--expert-cache-mib` that would stay under it: Basalt reports WDDM capping page-lock registration (`cudaHostRegister`) there, but NInfer pins with `cudaHostAlloc`, which may still succeed (full mode pins about 35 GiB on a 64 GB Windows machine). If the pin fails, startup stops with an error naming `--vram-headroom-mib` and `--partial-ram-residency`. Startup also refuses when the expert cache comes out smaller than planned. Not yet checked or measured on a GPU | off (every expert pinned) |
| `--expert-cache-mib N` | cap the expert cache (measurement) | what fits |
| `--shared-expert-stream` | run each MoE block's shared expert on a second CUDA stream beside the routed experts (high priority in eager calls only; CUDA graph replays ignore stream priority). Designed to leave the output bit for bit unchanged and to work with two lanes, MTP and vision; neither the identity nor the speed has been checked on a GPU yet | off |
| `--mtp-constrained` | keep MTP speculation on for constrained requests (JSON schema, grammar, tool calls): the verification accepts each draft under the grammar mask of its own position, as the Qwen3.5 models do, so greedy requests are designed to get the ids plain constrained decoding gives and sampled ones the same masked distribution. Needs `--spec mtp` (refused otherwise). Works with two lanes (a constrained request speculates while it decodes alone, like any request) and with vision. Not yet checked or measured on a GPU | off (constrained requests decode without MTP) |
| `--mtp-joint-acceptance` | choose how many MTP drafts a round verifies by Basalt's joint-acceptance rule instead of the acceptance estimate: each round drafts `--draft-tokens` (fewer near the request's end) and the drafter also returns each draft's probability; the round verifies the longest prefix whose estimated chance of being accepted whole stays above a threshold that adapts per request, and always at least the first draft. Only what is proposed changes, so the output is designed to stay what MTP gives without it. Needs `--spec mtp` (refused otherwise). Works with two lanes (while a request decodes alone, like any MTP round), vision and `--mtp-constrained` (the grammar masks cover the verified drafts). Every round pays for the full draft chain. Not yet checked or measured on a GPU | off (the acceptance estimate picks the draft count before drafting) |
| `--ngram-prefetch` | read the anchor column's n-gram rows into the volume's host cache while the MTP draft graph runs, so verification's row read finds them cached instead of waiting on the drive; the drafts' own rows are unknown until drafting ends and are read as before. Only when the host reads them moves, so the output is unchanged. Needs `--spec mtp` (refused otherwise); works with two lanes, vision, `--mtp-constrained` and `--mtp-joint-acceptance`. A round whose rows are already cached gains nothing. Not yet measured on a GPU | off (verification reads every row) |
| `--hot-embedding-rows N` | keep only the token embedding's first N rows (the lowest ids, which the tokenizer gives the most frequent tokens) in VRAM, as Basalt does with 8192: the whole table (322 MiB) moves to pinned RAM, the GPU reads the other rows from it over PCIe, and the VRAM saved goes to the expert cache (at 8192, 311 MiB, about 220 more experts). Designed to give the same embedding values, bit for bit, and to work with two lanes, MTP (the drafter takes the same route) and vision. Startup refuses N above the vocabulary (248,320), and any N with a token embedding that is not IQ4_XS (the GSQ-RCO IQ3_XXS pack's is IQ3_S). Not yet checked or measured on a GPU | `0` (the whole table in VRAM) |
| `--gdn-head-map` | read each GatedDeltaNet layer's 16 key heads for its 48 value heads in place, through the GGUF's tiled head map (value head h reads key head h mod 16), instead of first copying them to one key head per value head (216 copy nodes a decode token); MTP's GDN replay records then hold 16 key heads instead of 48. The same values reach the same arithmetic, so the output is designed to stay bit for bit unchanged; works with two lanes, MTP and vision. Not yet checked or measured on a GPU | off (the copies) |
| `--moe-gate-up-registers` | compute the routed experts' gate/up rows in a grid that stays resident on the GPU, each warp loading one gate and one up row straight into registers instead of staging 16 rows per thread block in shared memory, with the last sub-blocks of the two rows split across the warp's halves (Basalt's gate/up kernel). The same arithmetic in the same order, so the output is designed to stay bit for bit unchanged. Needs `--moe-precast-input` (refused without it); works with two lanes, MTP and vision. Not yet checked or measured on a GPU | off (rows staged in shared memory) |
| `--moe-precast-input` | cast each routed-expert call's input to A8 once (into the call's workspace, about 2.8 KiB a column) instead of in each of the 40 gate/up thread blocks per expert that read it. The same codes, so the output is designed to stay bit for bit unchanged; works with two lanes, MTP (the drafter's and the verification's calls too) and vision. The wide route of a prefill call casts its own copy as before. Not yet checked or measured on a GPU | off (each block casts its columns) |
| `--moe-fused-route` | route and dispatch the routed experts of each call of at most 8 columns (decode steps, MTP drafting and verification) in one kernel launch instead of two (one thread block picks each column's top 10 of 512 experts with the same code as before, then groups the entries by expert), with each expert's columns in ascending order rather than the order atomics happen to give. The same routing and the same arithmetic, so the output is designed to stay bit for bit unchanged; wider calls (prefill chunks) take the two launches as before. Works with two lanes, MTP and vision. Not yet checked or measured on a GPU | off (two launches) |
| `--cpu-verify-misses` | in each MTP verification round, compute some of every layer's routed experts that are not in the VRAM cache on the CPU instead of reading them over PCIe: per layer up to 4 of them, the ones with the fewest columns, with a third of the misses left on the GPU, computed by a team of CPU threads while the GPU runs the other experts and the shared expert. The CPU's arithmetic is the GPU's bit for bit, so the output is designed to stay unchanged. Plain decode and prefill are unchanged (a CPU leg cannot win at one column). Needs `--spec mtp` (refused otherwise); works with two lanes (while a request decodes alone, like any MTP round) and vision. The GPU waits for the CPU by spinning on a host flag inside the round; on Windows with a display attached a long wait could trip the driver's 2 s watchdog (TDR), A CPU that stops answering for 1 s ends the request with an error (nothing from that round is committed); later requests use the CPU again unless the CPU service itself has failed, in which case MTP is off for the rest of the run. The CPU threads spin while decoding, and after an idle gap the first answer can be late by the operating system's sleep granularity (on Windows up to about 15 ms). Not yet checked or measured on a GPU | off (misses are read over PCIe) |
| `--prefill-cpu-share F` | in prefill (and forced-token) calls, compute the share F of every layer's routed experts that are not in the VRAM cache on the CPU while the GPU copies the rest over PCIe and runs them: of a layer's n such experts, F × n rounded (halves up), the ones routed the fewest tokens first and none routed more than 8, on the same CPU threads as `--cpu-verify-misses` (`--cpu-verify-workers`). This reaches every prompt: one under 512 tokens (or the tail of one) is staged per call after routing, and one of 512 or more (a 1K-4K prompt is one 4,096-token call) copies its experts to the GPU ahead of time, so it leaves out the share F of each layer's experts that the cache has seen used least, and the CPU computes every one of those the prompt routes to at most 8 tokens (the GPU copies the others during the call). F is a decimal in [0, 1] with at most two places (Basalt uses 0.6); more places are refused rather than rounded; 0 is off. The CPU's arithmetic is the GPU's bit for bit, so the output is designed to stay unchanged. Needs no MTP; works with two lanes and vision, and beside `--cpu-verify-misses` (one CPU service serves both). Its channel pins about 40 MiB of RAM. The GPU waits for the CPU by spinning on a host flag inside each layer; on Windows with a display attached a long wait could trip the driver's 2 s watchdog (TDR). A CPU that stops answering for 1 s ends the request with an error; if the CPU service itself has failed, prefill stays on the GPU for the rest of the run. Calls wider than 4,096 tokens (`--prefill-chunk` above 4096) stay on the GPU. Not yet checked or measured on a GPU | `0` (off) |
| `--cpu-verify-workers N` | the CPU threads of `--cpu-verify-misses` and `--prefill-cpu-share`, the service thread included (1..64; needs one of them). PR 11 measured one expert alone (GPU idle) at 119 / 98 / 146 µs on 6 threads and 67 / 70 / 84 µs on 16 (IQ2_S / IQ2_XXS / IQ1_M); not measured inside a round | `6` |
| `--gdn-async-commit` | fold each MTP verification's accepted columns into the GDN recurrent state (the commit) on a CUDA stream of its own instead of the main one, so it runs beside the drafter's catch-up, the next round's drafts and the host's work between rounds; the next round's forward, and anything else that reads or writes the state (a prefill, a two-lane round, a context-cache capture or restore, the end of a request), waits for it first. A commit that ends the request folds on the main stream before the round is reported. The same fold in the same order, so the state is designed to stay bit for bit unchanged. Needs `--spec mtp` (refused otherwise); works with two lanes (while a request decodes alone, like any MTP round) and vision. No memory. Not yet checked or measured on a GPU | off (the commit folds on the main stream) |
| `--hc-int8` | read the hyper-connection mixers' weights (the 97 mixers that combine the residual streams before each block and at the end, 100 with the MTP drafter) from int8 copies with one scale per 32 weights instead of BF16: about 0.53 times the bytes each mixer reads per decode step, where the mixers are the largest single cost (about 2.8 ms of a ~12.5 ms token, from PR 6's per-call measurement of the BF16 mixer). Decode steps, MTP drafts and verification (up to 16 columns per call) read the copies; prefill keeps reading the BF16 weights, unchanged bit for bit. The copies are made at startup from the BF16 weights, which stay loaded, so it takes about 0.68 GB more VRAM (that much less expert cache). **Changes the numerics**: each weight is rounded to its int8 step, so the output is not bit for bit the same; activations stay in FP32. Judge it with the decode-path KLD gate (`--hc-int8` in the forward harness's `--kld-base` mode, [flash-next-plan.md](maintainer/flash-next-plan.md#decode-path-kld-gate)) before relying on it. Works with two lanes, MTP and vision. Not yet checked or measured on a GPU | off (BF16 weights) |
| `--hc-fused-read` | with `--hc-int8`: read each MoE block's input in one fused step on decode steps, MTP drafts and verification (up to 16 columns per call): the attention block's injection into the residual is applied as the int8 mixer reads it, and the router logits, the top-10 routing, the dispatch of the routed experts and the A8 cast the routed experts read are written by one kernel after the mixer's two. That is three launches per block instead of six (five with `--moe-fused-route`, one more with `--moe-precast-input`), and one pass fewer over the residual. The mixer kernels themselves are unchanged: a single persistent mixer kernel was measured slower on the RTX 3090 (PR 11). The same output as `--hc-int8` alone, bit for bit (dispatched experts in ascending entry order, as with `--moe-fused-route`); prefill is unchanged. With `--shared-expert-stream` the shared expert starts once the mixer's output is written. The shared expert's own input cast is not shared. Refused without `--hc-int8`. Works with two lanes, MTP and vision. Not yet checked or measured on a GPU | off |
| `--ngram-flag-wait` | run each decode round of a request decoding alone (and each MTP verification) as one CUDA graph instead of two. Today the layers before the first n-gram (PLE) layer run as one graph while the CPU reads the round's n-gram rows from the volume, then the CPU uploads them and launches the rest as a second graph. With this flag the whole round is launched at once: a one-thread kernel before the PLE layer waits on the GPU until the CPU has read the rows into pinned memory and set a flag, then the graph copies them in and runs the rest. The same kernels on the same data, so the output is designed to stay bit for bit unchanged. The first plain round and the first verification of each width on each lane, once per server run, still run as before (they set the kernels up). Two-lane rounds (two requests decoding together), `--no-cuda-graph` and `CUDA_LAUNCH_BLOCKING=1` keep the two parts (with launches serialized, as also under compute-sanitizer, a round waiting for the CPU would always time out: do not combine this flag with compute-sanitizer). The GPU waits by spinning on a host flag inside the round; on Windows with a display attached a long wait could trip the driver's 2 s watchdog (TDR). If the CPU does not set the flag within 1 s the request ends with an error (nothing from that round is committed). PR 11 measured +1.2% for the two-graph split this replaces; the one-graph form is not yet checked or measured on a GPU | off (two graphs around the row read) |
| `--max-concurrency 1\|2` | requests that decode in the same rounds (two lanes: see below) | 1 |

Two lanes (`set NINFER_CONCURRENCY=2`, opt-in): a second request no longer waits for the first. Two requests
decode in the same rounds (plain, without MTP: speculation runs only while one request is decoding), and a lane's
prefill calls run between the other lane's decode rounds, so it pauses for up to one call (about 1.7 s per
4,096-token call). On the spike's eight-conversation agent burst it finished 6.5% sooner and cut the mean time to
first token by 19%, which is less than the 1.15x aggregate decode the design aimed for
([measurements](maintainer/flash-next-plan.md#pr-16-two-lanes)). One lane stays the default; three or more are refused.

### A default expert state

The saved state (`<model>.expert-state`) is tied to the copy of the model that wrote it (its path,
size and modification time), so a fresh install, or a re-downloaded model, starts with an empty
cache. A default state is tied to the artifact id instead, which every copy of one conversion
shares, so it can ship with the model and warm the first start. The server loads files in this
order: the saved state, when it exists and belongs to this copy of the model; else the default
(`<model>.expert-state.default`, or `--default-expert-state FILE`), when it exists and its artifact id
matches; else it starts empty. The default is only ever read: the server saves to the saved state
beside it, which takes over from the next start. A default of another model, another expert count
or a damaged one is skipped with a line in the log. `--no-expert-state` starts empty and does not
read the default. No default ships with any model yet, so nothing changes until one is present.

To make one for a release, run a representative session so the saved state fills, stop the server,
and rewrite the saved state with the artifact's identity (the tool is built with the apps, in
`build/tools/`):

```
ninfer-qwen4-exp-expert-state-export <model>.expert-state <model>.expert-state.default <model>.ninfer
```

It reads the artifact id from the model and refuses a saved state written for another copy; pass
`--artifact-id <32 hex digits>` in place of the model path to skip that check. Not yet measured on a
GPU: how much a default warms the first start.

### Vision residency

`--vision` keeps the Vision tower, its encode workspace and the item handoff resident for the
process lifetime. `--vision-residency overlay` removes that cost on memory-tight cards: the tower
lives in pinned host memory, the sequence plan reserves nothing for Vision, and each image is
encoded inside a bounded window whose device memory is borrowed for the duration of the encode.

A window is funded from free KV pages when they cover it. Those pages hold nothing, so nothing is
copied, the text weights stay mapped, and the encode runs on its own stream while other lanes keep
decoding: the window is opened as soon as the image's prefill unit is licensed, after every other
lane's unit for that round, and the lane yields its prefill turn until the encode completes. When
free pages fall short, cached context is reclaimed (demoted or released) to make room; the window
never revokes a paused request's snapshot or pauses another request. The pages are out of
circulation while the loan is open, and any request that then finds itself short of KV pages first
waits for the encode to finish and the pages to return, so a loan never causes an eviction or a
preemption of its own.

When free pages cannot cover the window, it falls back to the evict-ranked tail of the text weights
(lm_head, token embedding, draft head, MTP head), restored from a pinned mirror before the prefill
unit ends. That window is exclusive: the borrowed weights are unmapped, so nothing else runs until
it closes. The fallback is always available, so a vision request never fails for lack of memory,
and a KV cache smaller than one window simply uses it for every image.

Either way the window streams the tower through two layer slots, lands the embeddings in pinned
memory, and the prefill then uploads only each chunk's embedding columns. A follow-up turn over the
same image reuses the prefix and opens no window. Embeddings are produced by the same kernels on the
same bytes, so completions are identical between residencies. `--vision-max-merged N` bounds one
item's merged tokens (media above it is downscaled at preprocessing) and sizes the window. The
request log line reports `overlay=<windows>x<conc|excl|mixed>[+ahead] <ms> (evict <MiB> <ms>,
restore <ms>, staged <MiB>)`, where `+ahead` marks windows that encoded beside other lanes' work,
and the JSON record carries `vision_overlay`, including `exclusive_windows` and `ahead_windows`.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health and build version (see [Server version](#server-version)) |
| `GET /v1/load` | serving capacity, current load, and monotonic token counters (see [Load](#load)) |
| `GET /slots` | each execution lane's state and progress, in llama.cpp's array shape (see [Slots](#slots)) |
| `GET /metrics` | Prometheus counters, gauges and latency histograms, plus llama.cpp-compatible series (see [Metrics](#metrics)) |
| `GET /dashboard` | a live dashboard page for a browser, drawn from `/metrics`, `/v1/load` and `/slots` (see [Dashboard](#dashboard)) |
| `GET /v1/models` | configured OpenAI model alias, effective context limit (`max_model_len`/`context_window`/`context_length`) and input modalities (see [Model discovery](#model-discovery)) |
| `GET /v1/models/{id}` | lookup of the same model object by its alias |
| `GET /props` | read-only llama.cpp-style server properties (see [Model discovery](#model-discovery)) |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. On Linux, accepted
connections also use TCP keepalive and a 15-second `TCP_USER_TIMEOUT`; together with the heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.

The HTTP transport is thread-per-connection, and a worker stays with a connection for its whole
life, including the idle window between keep-alive requests. The pool therefore reserves one base
worker per admissible request (`max_concurrency + max_pending_requests + 1`) and grows on demand up
to 64 further workers for connections that are merely open; the extra workers retire once idle.
Without that headroom a client-side connection pool of otherwise idle sockets occupies every worker
and the server accepts real requests strictly one at a time.

### Model discovery

`/v1/models` lists one model object:

```json
{"id": "qwen3.8-27b", "object": "model", "created": 1790000000, "owned_by": "ninfer",
 "max_model_len": 65536, "context_window": 65536, "context_length": 65536,
 "architecture": {"input_modalities": ["text", "image", "video"], "output_modalities": ["text"]}}
```

The context limit is `--max-context`, mirrored under the vLLM/llama.cpp, Anthropic, and
OpenRouter/Ollama field names. `architecture` uses the OpenRouter shape that llama.cpp's router-mode
`/models` also emits; `input_modalities` lists `image` and `video` only when the server runs with
`--vision`, and is `["text"]` otherwise.

`GET /props` serves clients that discover a llama.cpp server. It is read-only and fills only the
llama.cpp fields NInfer can state truthfully:

```json
{"default_generation_settings": {"n_ctx": 65536,
   "params": {"n_predict": -1, "max_tokens": -1, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
              "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}},
 "total_slots": 1, "model_alias": "qwen3.8-27b", "model_path": "models/qwen3_8_27b.ninfer",
 "modalities": {"vision": false, "audio": false}, "build_info": "ninfer 0.14.3-rtx3090"}
```

`n_ctx` is `--max-context` and `total_slots` is `--max-concurrency`. `n_predict` and its alias
`max_tokens` are the [default output limit](#default-output-limit): `-1`, llama.cpp's "no fixed
cap", when a request may run to the remaining context, or the `--default-max-tokens` cap. `--max-output-tokens` bounds either, so it is reported instead when it is smaller or when
no default is set: the largest budget any request can get. The sampler is the loaded model's preset
for the default thinking mode (thinking unless `--no-thinking`) under the process sampling flags and
`--greedy`; request fields still override it per request. `seed` appears only with `--seed`, since
requests otherwise draw a fresh random seed. `model_alias` is the public model id and `model_path`
the artifact path the server was started with. `build_info` is `ninfer <version>` (see
[Server version](#server-version)). There is no `chat_template` or
writable `POST /props`, and llama.cpp's non-`/v1` route aliases are not served. `GET /slots`
follows llama.cpp's array shape (see [Slots](#slots)).

### Server version

The running build is reported four ways, all from the same string: `ninfer-serve --version` (and
`ninfer --version`) print it and exit without loading a model; `GET /health` returns it as
`{"status": "ok", "version": "0.14.3-rtx3090"}`; every response, including the `503` during model
load and `401` failures, carries an `X-NInfer-Version` header; and the `server_start` log record
and `/props` `build_info` include it.

The string is the repository's `VERSION` file. A build from any other commit appends
`+<9-char commit>` (`0.14.4-rtx3090+1a2b3c4d5`), and uncommitted changes to tracked files append
`-dirty`, so a development binary is never mistaken for the release it descends from. A build
without git metadata, such as a `git archive` snapshot, reports `VERSION` alone, which is why a
release must be built after `VERSION` is set.

### Startup readiness

The server binds its port and starts accepting connections before the Engine has loaded weights
and finished warmup, so a port clash is reported in milliseconds rather than after loading. Every
route -- including `/health`, `OPTIONS`, and requests that would otherwise be unauthenticated --
answers `503` with `Retry-After: 2` until warmup completes, in the target's own error shape (only
the static [dashboard](#dashboard) page is served meanwhile, and it shows the `503` its polls get):

```json
{"error":{"message":"The model is still loading. Retry shortly.","type":"service_unavailable","code":"model_loading"}}
```

`POST /v1/messages` and `/v1/messages/count_tokens` receive the Anthropic envelope instead, with a
`request_id` field and a `request-id` header, matching every other error response on those
endpoints:

```json
{"type":"error","error":{"type":"api_error","message":"The model is still loading. Retry shortly."},"request_id":"req_..."}
```

A readiness probe should poll `GET /health` (or any endpoint) and expect `503` until the model is
ready rather than treating an accepted TCP connection as a signal of readiness.

After startup, an internal host-side failure fails only the requests that were running or being
admitted, clears the context cache, and keeps serving: queued requests stay queued, and a paused
request keeps its place but gives up its saved snapshot and resumes by recomputing its prompt and
output. Each such recovery is counted in `ninfer_engine_recoveries_total` and as a top-level
`engine_recoveries` counter on the request log's `throughput` event, and `GET /health` stays `200`.
A failure confined to admitting one request (planning it or starting its binding) fails that
request alone and is not a recovery. `GET /health` turns `503` for good only when the engine cannot
prove a clean recovery (cleanup fails, or the GPU memory and host cache in use do not return exactly
to their state after startup), when the GPU itself reports an error, or after three failures in a
row (each less than 30 seconds after the previous one, with no request completing between them);
the server then needs a restart. Every such failure is logged at error severity with the exception
text, the execution unit (`boundary`, `admission`, `control`, `prefill` or `decode`), the affected
request ids and lanes, and the streak count; a latch is logged `FATAL`, and `ninfer-serve` exits
with status 3 after a five second grace period (so in-flight error responses flush and `/health`
answers 503 meanwhile) for a supervisor to restart. `--no-exit-on-engine-failure` keeps the
process alive instead, answering `503`. The exception text is operator diagnostics and is never
sent to clients; the request log's `request_error` record carries it for requests that failed this
way.

### Load

`GET /v1/load` is a cheap, pollable snapshot for load balancers and gateways that schedule across
several servers. It requires the API key when one is configured, answers `503 model_loading` until
warmup completes like every other route, and reads only the Engine's already-published runtime
counters and the ingress count, so polling it does not wait on or delay the GPU executor.

```bash
curl http://127.0.0.1:8080/v1/load -H 'Authorization: Bearer local-secret'
```

```json
{
  "object": "ninfer.load",
  "model": "qwen3.8-27b",
  "uptime_seconds": 812.4,
  "capacity": {"max_concurrency": 4, "max_pending_requests": 32, "max_admitted_requests": 36,
               "max_context": 65536, "kv_capacity_tokens": 131072, "kv_capacity_pages": 2048,
               "kv_page_tokens": 64, "device_state_slots": 8, "host_context_bytes": 17179869184},
  "requests": {"admitted": 6, "running": 4, "prefilling": 1, "decode_ready": 3, "waiting": 2,
               "paused": 0, "replaying": 0, "materializing": 0, "prefill_backlog_seconds": 14.2,
               "queue": [{"id": 311, "prompt_tokens": 18240, "waited_s": 4.1},
                         {"id": 312, "prompt_tokens": 96, "waited_s": 0.3}],
               "queue_total": 2},
  "occupancy": {"device_main_kv_pages": 1500, "device_main_kv_tokens": 96000,
                "device_state_slots": 5, "host_context_bytes": 2684354560,
                "host_state_slots": 7, "host_kv_bytes": 2147483648},
  "counters": {"computed_prefill_tokens": 48200113, "committed_decode_tokens": 6120452,
               "reused_prompt_tokens": 30911840, "decode_rounds": 861307,
               "decode_row_rounds": 2448180},
  "host": {"active_seconds": 3120.4, "device_wait_seconds": 41872.9},
  "last_request": {"completion_tokens": 24, "generation_wall_seconds": 0.17,
                   "inter_token_seconds": 0.0074, "decode_host_seconds": 0.012}
}
```

- `uptime_seconds` counts from the moment the server became ready, not from process start.
- `capacity` is fixed once the Engine is ready. `max_admitted_requests` is
  `max_concurrency + max_pending_requests`, the ingress bound described in
  [Execution behavior](#execution-behavior); `kv_capacity_tokens` is the resolved page-aligned Main
  KV pool and `kv_page_tokens` its page size.
- `requests.admitted` counts requests holding ingress capacity, from preparation until the response
  is released; a new generation request is rejected with `server_overloaded` (HTTP 429, or 529 on
  Anthropic endpoints) when it would exceed `max_admitted_requests`. `running` counts occupied
  execution lanes (at most `max_concurrency`), of which `prefilling` counts the lanes with prompt
  left to prefill, `replaying` the lanes rebuilding a paused request's state, and `decode_ready` the
  lanes in the decode batch. `waiting` counts requests submitted to the Engine FIFO that have not
  been admitted to a lane yet. `paused` counts admitted requests that resource pressure paused; they
  hold no lane and resume in their original order. `prefill_backlog_seconds` is the prefill time the
  Engine's context-cost model predicts is ahead of a new request: what the resident prefills have left
  plus what each queued request costs from scratch (a cache hit makes it cheaper). It leaves out decode
  and paused requests and is 0 when idle, so a gateway choosing between servers can prefer the smaller
  backlog; it is a prediction from the machine's fitted costs, not a promise.
- `requests.queue` lists the waiting requests, oldest first: the Engine request `id` (the
  [structured request log](#structured-request-log)'s `engine_request_id`), its `prompt_tokens`
  and the seconds since it was submitted (`waited_s`). It lists at most the first 64;
  `queue_total` counts them all and equals `waiting`. Paused requests are not in it.
- `capacity.host_context_bytes` is the one pinned Host budget (see
  [Automatic host cache](#automatic-host-cache)). `occupancy` reports current Main KV pages (and
  tokens), Device StateImage slots, and the Host context bytes in use, including destinations
  reserved for transfers in flight; `host_state_slots` and `host_kv_bytes` break that Host figure
  down and must not be added to it. Main KV occupancy includes retained reusable prefixes, which the
  scheduler reclaims under pressure, so a full pool does not by itself mean new requests will wait.
- `counters` are monotonic since startup; derive rates by differencing two polls.
  `computed_prefill_tokens` excludes prefix-reused prompt tokens (reported separately in
  `reused_prompt_tokens`). `committed_decode_tokens` counts tokens committed by decode rounds and
  excludes each request's first token, which prefill emits; with speculative decoding a round commits
  several tokens per row. `decode_row_rounds` is the sum of decode batch sizes over `decode_rounds`.
- Gauges and counters come from the snapshot the Engine publishes at execution boundaries, so they
  can trail the instant of the poll by up to one boundary.
- `host` is cumulative Engine host-active seconds since startup (device wait excluded) and,
  separately, seconds the host thread waited on the device. Take two polls and divide the host delta
  by the interval to see how host-bound the Engine is.
- `last_request` is the pace of the last finished request that produced at least two tokens, or
  `null` before the first. `inter_token_seconds` is first-to-last token wall time over the token
  gaps. A healthy Engine stays in milliseconds; an Engine that stalls for seconds right after every
  first token reports whole seconds even for a short answer while `/health` stays 200, so a
  supervisor can restart on, say, `inter_token_seconds > 0.25` for several consecutive polls.

### Slots

`GET /slots` shows what each execution lane is doing, for clients that probe a llama.cpp server's
`/slots` and for watching a long prefill. Like `/v1/load` it requires the API key when one is
configured, answers `503 model_loading` until warmup completes, and reads only the snapshot the
Engine already publishes at every prefill step and decode round, so polling it never waits on the
GPU executor. The body follows llama.cpp's shape, a top-level array with one object per slot
carrying llama.cpp's `id`, `is_processing` and `n_ctx`, and adds NInfer's progress and
speculative-decoding fields:

```json
[{"id": 0, "is_processing": true, "state": "prefill", "request_id": 7, "n_ctx": 65536,
  "prompt_tokens": 4096, "prompt_done": 3072, "reused": 2048, "generated": 0, "n_ctx_used": 3072,
  "tokens_per_second": 0.0, "drafted": 0, "accepted": 0, "relaxed_kept": 0},
 {"id": 1, "is_processing": true, "state": "decoding", "request_id": 5, "n_ctx": 65536,
  "prompt_tokens": 100, "prompt_done": 100, "reused": 0, "generated": 41, "n_ctx_used": 141,
  "tokens_per_second": 62.5, "drafted": 48, "accepted": 30, "relaxed_kept": 2}]
```

- There is one entry per execution lane (`--max-concurrency`). `id` is the lane, and `n_ctx` is
  `--max-context`, the context any one request may use. `request_id` is the Engine request id that
  the [structured request log](#structured-request-log) records as `engine_request_id`, or `null`
  on an idle lane, whose counts are all 0.
- `state` is `idle`, `prefill` (the lane is prefilling its prompt, or rebuilding a paused request's
  state) or `decoding` (the prompt is done and the lane is producing tokens); `is_processing` is
  `state != "idle"`. A lane being paused reports the phase it was in: the rebuild it interrupted,
  or else the phase it resumes in.
- `prompt_tokens` is the request's prompt. `prompt_done` is how much of it the lane holds so far,
  `reused` (the prefix taken from the context cache) included, so `prompt_done / prompt_tokens` is
  the prefill progress and the two are equal once prefill is done. `generated` counts committed
  output tokens, the first included, and `n_ctx_used` is `prompt_done + generated`. While a lane
  rebuilds a paused request's state, `prompt_done` and `n_ctx_used` stay where they were: the
  replayed tokens are counted separately (`replayed_tokens` in the request log).
- `tokens_per_second` is the lane's tokens after the first over the time since the first, so it
  includes any pause; it is 0 until the second token.
- `drafted`, `accepted` and `relaxed_kept` are the request's speculative-decoding counts so far:
  draft tokens offered, drafts accepted, and the accepted drafts that
  [relaxed drafting](#relaxed-drafting) kept (a subset of `accepted`). All are 0 without a draft
  source.
- Requests waiting for a lane are not slots: `/v1/load` lists them under `requests.queue`.
- The snapshot can trail the poll by up to one prefill step or decode round, so a long decode round
  or prefill chunk shows as a pause in the counts.

### Dashboard

Open `http://127.0.0.1:8080/dashboard` (your `--host` and `--port`) in a browser for a live view of
the server. It is one page built into `ninfer-serve`, with nothing fetched from elsewhere, so it
works offline. Once a second while its tab is visible (it stops polling when the tab is hidden) it
reads `/metrics`, `/v1/load` and `/slots` from the same server and shows:

- the current decode and prefill tokens/s, and charts of both over the last two minutes, from
  `ninfer_decode_tokens_total` and `ninfer_prefill_tokens_total` (prefix-cache hits excluded);
- draft acceptance: a chart of each second's accepted / drafted tokens, and the figure since startup
  (`off` without a draft source);
- running and waiting requests, and the share of the Main KV pool in use (retained prefixes
  included, as in `/v1/load`'s `occupancy`);
- GPU power (with a chart), utilization, VRAM in use by every process on the card, and core
  temperature, from the `ninfer_gpu_*` gauges (`-` when NVML gives no reading);
- one row per lane from `/slots`: its state, request, prompt progress with the reused prefix,
  generated tokens, context use against `--max-context`, tokens/s and accepted drafts; and below
  them the requests waiting for a lane, from `/v1/load`'s `requests.queue`.

The history lives in the page, so a reload starts the charts empty. Rates are differences between
two polls of the counters, and the counters advance at prefill steps and decode rounds, so a second
can read high or low next to its neighbours.

**The key.** The page itself is served without `--api-key`, since it holds no data, but everything
it polls needs the key. If the server has one, the page says so: type the key into the field at the
top and press Enter or **Use key**. The page sends it as `Authorization: Bearer` on its own
requests and keeps it in the tab's session storage, so it survives a reload of that tab but not
closing it; **Forget** clears it. The key never appears in the URL. The
[Host check](#host-and-origin-checks) applies to the page like every route, so open it by a name
the server answers to; its requests go to its own origin, so they pass the Origin check.

The page is served with a `Content-Security-Policy` that lets it run only its own inline script and
style and connect only to its own origin, forbids framing it, and with
`X-Content-Type-Options: nosniff`. Under the [router](#several-models-router) the router serves the
page itself, and what it polls answers for the model that last served.

### Context store

`--context-store DIR` makes the context cache survive a restart or a crash. It is off by default
and needs somewhere to hold checkpoints: a Host context cache (the default 8 GiB,
`--host-context-mib` or `--auto-host-cache`), or a disk tier (`--context-nvme-mib N`, which also
works with `--host-context-mib 0`), because stored sessions come back into that tier. What is stored is one retained conversation
(continuation) at a time: its recovery points (the endpoint of the last turn, the point where the
next turn's input starts, any long anchors) with their KV pages and recurrent states, byte for byte
what the Host tier would hold for them. Public shared prefixes (a system or tools prompt that the
cache keeps as one entry for every conversation that starts with it) are stored too, each as an
image of its own, written as soon as the cache has captured one (at most one a second, while the
writer is idle, busy or not: it is a short copy of a few thousand tokens and one recurrent state,
counted in `ninfer_context_store_shared_export_seconds_total`), so after a restart, or on a new
box that shares the bucket, the first request of every tenant on that prompt starts warm. Prompt
grafts are installed context and are never stored. A conversation is written to `DIR`

- in the background once it has been unused for `--context-store-idle-seconds` (30 s) and has
  changed since it was last written, so a crash loses at most that much of a conversation,
- when the cache is about to evict it (write-on-evict): a conversation not idle long enough to have
  been written, or changed since, is copied out and queued for the store just before its points are
  released, so it comes back by hydration instead of a full prefill, and
- at shutdown, for every conversation not already stored, most recently used first, within
  `--context-store-flush-seconds` (60 s).

With `--context-store-idle-seconds 0` there are no background writes; eviction and shutdown still
write. The cache's eviction knows what the store holds: a conversation the store holds as it is now
is priced as the time to read it back and copy it to the GPU rather than the prefill it would
otherwise cost, so under pressure it goes before one that would have to be recomputed (when the
store is slower than prefilling it, nothing changes).

On start-up the stored images are restored into the Host tier while they fit in its free space and
until `--context-store-restore-seconds` (120 s) is spent, before the server accepts requests, in
order of value: shared prefixes first, then conversations by the prefill time they save per byte of
Host tier they take (the cost model's prefill estimate for their length, over their image size),
the most recently used first among equals. A deep conversation therefore goes before a short one
that would fill the same room, and what does not fit stays in the store for hydration. Nothing is evicted for them. A request then reuses a restored
conversation exactly as it would have before the restart, restoring it from the Host tier to the
GPU like any other Host-resident checkpoint. There is nothing for a supervisor or gateway to call.

The store is also read while the server runs. When a request is about to be admitted and the store
holds a checkpoint of that very prompt deeper than the deepest checkpoint the cache offers it (the
conversation was evicted since), and reading that image back, copying it to the GPU and prefilling
the rest is cheaper than the best start the cache offers, a background reader loads the image while
the request waits. The context cost model prices all three: the prefill and copy terms are the
calibrated ones the cache already ranks its own sources by, and the read is a fixed 5 ms plus the
image size at a rate that starts at 1 GB/s and is refined from every read the server makes (at
start-up and for requests), so a slow or a fast store volume is priced as it is
(`ninfer_context_store_read_bytes_per_second` shows the rate in use). A short gain on a large image
is therefore prefilled, and a long one is read back however deep the cache's own start is. The
reader runs out of the admission order but without holding up other requests, for at most
`--context-store-restore-seconds` or the request's queue deadline. The image is then rebuilt in the
Host tier, giving up the cache's lowest-ranked Host contents if it needs the room (never another
waiting request's chosen source), and the request resumes from it instead of prefilling the
difference. A stored conversation that cannot be read, does not fit or misses its deadline is a miss,
and the request is prefilled as without the store. A read counts as a hydration only if the
admission actually resumes from it. On the 27B on an RTX 3090, a 7.8k-token conversation (an
856 MB image) read back from a warm file cache in about 1.5 s gave a first token after 1.6 s, against
5.1 s for prefilling it and 0.1 s when it had stayed resident; that was before the store read its
chunks in parallel, which about tripled its read rate ([measurements](performance.md#context-store-read-and-write-speed-rtx-3090-host-windows-11-c-nvme)).

The store works with MTP and DFlash2, including the launchers' default DFlash2, and with no
speculation. A stored recovery point carries the draft's own context with the target's: DFlash2's
local K/V rings (40 MiB per recovery point on the 27B) inside each recurrent state. A restored
conversation therefore drafts exactly as if it had never left the GPU: the real tests require the
same output and the same speculative rounds and accepted drafts as the uninterrupted conversation.
`--spec dflash` (the 35B-A3B DFlash v1 backend, not a launcher profile) is accepted but is not
fully verified: after a restart a short conversation was restored at start-up yet its continuation
reused no tokens, so it falls back to a cold prefill (a deep conversation restores and matches
exactly), and its hydration, crash and recovery paths were not exercised. Treat the store as
unsupported there.
N-gram copy drafting (`--ngram-draft-tokens`) needs nothing stored; a restored request builds its
copy index from its own prompt like any other. A store is bound to the speculative backend, draft
count and proposal head it was written with (and to the model, KV format and `--gdn-state-fp16`), so
a server restarted with another `--spec` or `--draft-tokens` restores nothing from it and prefills
instead; the start-up log counts those sessions as written under another configuration, and they
age out. `--ngram-draft-tokens` may change freely.

An image does not depend on `--devices`/`--stage-layers`: a conversation stored by a server split
into pipeline stages restores on one GPU and the reverse. A worker failure that clears the context
cache leaves the store as it was, and the next turn of a stored conversation reads it back. Prompt
grafts are installed context, never stored; a conversation that started from a graft is stored like
any other, and changing the configured grafts (their names or files) makes the existing images
misses, as a different model would.

A session image is several GB for a deep context (about 18 KB per token with `--kv-dtype rk4v4`,
plus about 150 MB of recurrent state per recovery point, about 80 MB with `--gdn-state-fp16`, and
40 MiB more under DFlash2), and consecutive images of one conversation
share almost all of it. The store therefore splits an image into fixed 32 MiB chunks named by a hash
of their content and writes only chunks it does not already hold: keeping a long conversation current
costs the newest pages and the endpoint state, not the whole session. Older images of a conversation
are removed once a newer image of it is stored. Writes are atomic (a temporary name, then a rename),
every chunk is verified when read, and a session that cannot be read back intact is removed and
simply re-prefilled by the next request; a damaged store costs cache hits, never a wrong answer.

The store keeps at most `--context-store-max-gib` (default: half the free space of the volume when
the server starts), removing the least recently used sessions first, and removes sessions unused for
`--context-store-ttl-hours` (default 168). A store written by a different model, quantization, KV
or recurrent-state configuration, speculative configuration, set of prompt grafts or an older
engine's image format is ignored and ages out; the start-up log line `context store | N sessions
were written under another model, configuration or store format` says so. The `ninfer_context_store_*` series (see
[Metrics](#metrics)) report size, writes, bytes reused, what was restored at start-up and how long it
took. A background write is skipped while any request is waiting, binding, prefilling or replaying,
and the background queue holds at most two images, so a slow disk does not hold up requests; an
evicted conversation is queued while less than 4 GiB waits (always into an empty queue), in ordinary
memory, so its bytes outlive the Host copy the eviction releases until the writer has stored them.
Taking the image itself (copying a conversation out of the GPU or the Host tier) is a single step of
the Engine worker, so a request that arrives during it waits for that copy: at eviction it adds to
the admission that needed the room (`ninfer_context_store_eviction_export_seconds_total` counts
it). A conversation that could not be queued is tried again later if it is idle, or, if it was
being evicted, is lost to the store and prefilled on its next turn as without one. The cache marks
a conversation stored when its image is queued; a write that then fails, or an image the store
later removes for space or age, clears every mark, so everything is written again.

`--context-store-compress` (off by default) writes new chunks zstd-compressed (level 3), in the
directory and in the bucket, when that makes the chunk at least 3 % smaller (otherwise it is kept
raw). The saving is modest because the KV is already 4-bit: measured on stores written with
`--kv-dtype rk4v4` and 40-60k-token conversations, KV pages shrank by about 5-10 %, recurrent state
by 13-17 % (17 % with `--gdn-state-fp16`), and the whole store 8 % (3.75 GB raw, 3.45 GB compressed, same workload); it only saves disk
space and upload time, not VRAM or Host memory, which always hold raw bytes. A compressed chunk is a
`<hash>.chunk.zst` file (`.chunk.zst` object in the bucket) beside the raw `<hash>.chunk` form, still
named by the hash of its raw content, so manifests are unchanged and a store can hold both forms.
Chunks of either form are always read, whatever the flag says, and the store's size limit counts
what the files take on disk. Compression is a stream in both directions: a chunk is compressed into
its file, and a stored chunk is decompressed straight into the image being read, with no second copy
of either form (decompression runs on the same threads that read the chunks, at several hundred MB/s
each), which is not free: restoring the 3.75 GB store above at start-up took 4.2-4.4 s from
compressed chunks against 3.3-3.7 s from raw ones (file cache warm, RTX 3090 host), about 0.25 s
more per GB read. A chunk that does not decode to its hash is a miss like any other damaged chunk.


#### Two engines on one box

A box with two GPUs serves long conversations fastest as two independent engines, one per GPU (a
single engine split across both GPUs does not pipeline prefill). A gateway then sends a
conversation's next turn to whichever engine is free, often the one that did not serve the last
turn, which would prefill the whole conversation again. `--context-store-peer DIR` lets an engine
reuse what the other finished, through the local disk and without a bucket:

```
ninfer-serve model.ninfer --port 8000 --devices 0 --context-store /data/store0 --context-store-peer /data/store1 --context-store-idle-seconds 2
ninfer-serve model.ninfer --port 8001 --devices 1 --context-store /data/store1 --context-store-peer /data/store0 --context-store-idle-seconds 2
```

- Every engine has its own `--context-store` directory and is its only writer. A second engine
  started on a directory already in use refuses to start (`context store directory ... is in use by
  another process`) instead of corrupting it; the lock is released when the process exits, so a
  crash leaves nothing stuck. (Before this, two engines on one directory silently deleted each
  other's files.)
- `--context-store-peer DIR` (repeatable) names another engine's directory. Every
  `--context-store-peer-poll-ms` (250) the engine lists the manifests there, and for each session it
  does not hold it **hard-links** the session's chunk files into its own directory (no bytes are
  copied, the data is on disk once) and registers it like any stored session. A request that would
  resume from it finds it through the same lookup as its own sessions (see above), reads and
  verifies every chunk, and prices the read with the measured read rate. A peer that has not started
  yet is polled until it appears. Both directories must be on one volume: a link that fails (another
  volume, or a chunk the peer evicted in the meantime) skips that session, which is only a miss.
  With a bucket as well, a session the bucket listing registered before its chunks were downloaded
  takes the peer's chunks the same way, so loading it reads the disk instead of the bucket.
- An imported session is the importer's own from then on: the peer deleting, evicting or replacing
  its copy does not affect it, and removing it here does not affect the peer. It is not uploaded to
  the bucket by the importer (the engine that wrote it uploads it), until the importer restarts: its
  start-up scan cannot tell an imported session from its own, so from then on a use puts back what
  the bucket lacks, as for its own sessions. Sessions older than
  `--context-store-ttl-hours`, or that would not fit in `--context-store-max-gib`, are skipped, and a
  skipped session is not looked at again until its owner rewrites it.
- **A session is visible to the other engine only after its owner has written it**, which is
  `--context-store-idle-seconds` (30 by default) after its last use, or when it is evicted, or at
  shutdown, and a background write is skipped while any request is waiting. With the default, the
  next turn of a conversation that arrives within 30 s of the last one cannot be resumed on the other
  engine; use `--context-store-idle-seconds 1` or `2` on a box with peers. The cost is more writes
  (only the chunks that changed, since consecutive images of a conversation share almost all of
  theirs).
- Failures are extra misses, never wrong output: every chunk is verified when read, a damaged peer
  file is a miss for the importer and is never modified or removed by it, and a manifest is checked
  by its own checksum. The `ninfer_context_store_peer_*` series (see [Metrics](#metrics)) count what
  was imported, the bytes linked, sessions skipped and links that failed.

#### Keeping a copy in a bucket

`--context-store-s3-endpoint URL --context-store-s3-bucket NAME` (off by default, and only with
`--context-store`) keep a copy of the store in an S3-compatible bucket: AWS S3, MinIO, Cloudflare
R2, Backblaze B2 or any server that speaks the S3 API with path-style addressing and Signature V4.
The local directory becomes a cache of the bucket.

- Every session written to the directory is also uploaded in the background, chunks first and its
  manifest last, so another engine never sees a session whose chunks are missing. A chunk the bucket
  already holds is not sent again, so a conversation that grows uploads only what changed: its new
  KV pages, its new recurrent states (each saved state is chunked on its own, so one that did not
  change is never sent again wherever it sits in the session), and the few pieces of its per-token
  header that grew (cut by content, about 0.5 MB a turn). A manifest is a few KB. Conversations that
  start with the same shared prefix (an agent's system and tools prompt) share its chunks, here and
  in the bucket, so the prefix is stored and sent once. Shutdown waits up to two minutes for the
  uploads still queued.
- A written session waits `--context-store-s3-upload-delay-seconds` (default 30) before it is
  uploaded. Each turn's write supersedes the one before, and a superseded write still waiting is
  never sent, so a burst of turns (a short `--context-store-idle-seconds` writes several per
  request) costs one upload. A conversation that keeps going is still uploaded at least that often.
  Once a session is in the bucket, the earlier turns it superseded are deleted from it, so the
  bucket holds one session per conversation and a new engine lists only those.
- The bucket's keys sit under a folder named by a hash of the model, configuration and artifact
  (the store's binding), so an engine lists only sessions it could use.
- At start-up, and every minute after, the engine lists the bucket and registers sessions it does
  not have, reading only their manifests. They appear in the store as not local. Start-up restores
  only sessions the directory holds: nothing is downloaded on the chance a request might want it.
- Sessions pass between engines that load the same artifact (by the id its conversion wrote, so any
  copy of it, whenever it was downloaded) with the same KV, recurrent-state, speculative and vision
  settings. A session written under anything else is never read; it is a miss, and ages out. The
  exception is a configured prompt graft: it is matched by its file's size and modification time, so
  sessions that start from a graft pass between engines only when the graft file is the same copy.
- A request that finds nothing usable, and has more than about two seconds of prefill ahead, has
  the bucket listed once more before it prefills (waiting at most five seconds; requests that
  arrive while a listing is taken share the next one), in case another engine stored the session
  since the last listing: a conversation that just moved here is found.
- A request that would resume from a session only the bucket holds waits for it like any stored
  session (see above): the background reader downloads the missing chunks (several at a time, each
  verified; a damaged one makes the session a miss and it is not offered again), within the same
  deadline, without holding up the Engine worker or other requests. The download is not weighed
  against the prefill it saves, since only this request waits for it while a prefill holds the
  GPU for everyone, but it must fit in the wait at the rate fetches (download, verification and
  write together) have achieved (`ninfer_context_store_remote_fetch_bytes_per_second`). Chunks the
  directory already holds, such as a shared prefix's, are not downloaded. Listings for requests
  run on their own thread, so a slow bucket never holds up reading a stored session.
- A session evicted from the directory for space (or age) gives up its chunk files, except those an
  image still held whole shares, but stays in the index, since the bucket still holds it, and is
  fetched back the same way when it is needed. Its manifest is not downloaded again. Shared
  prefixes are evicted last, and every conversation written that starts with one counts as a use
  of it (here and in the bucket), so a prefix many conversations rely on is not evicted or expired
  as unused while they reuse it from memory.
- Credentials come from the environment, never the command line: `NINFER_S3_ACCESS_KEY_ID` and
  `NINFER_S3_SECRET_ACCESS_KEY` (or `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; an
  `AWS_SESSION_TOKEN` is honoured). `--context-store-s3-region` (default `us-east-1`) names the
  signing region and `--context-store-s3-prefix` namespaces the keys when a bucket is shared. Use
  an `https://` endpoint outside a trusted network: payloads are not part of the request signature.
- The bucket is the long-term tier, so its expiry is the bucket's: add a lifecycle rule that expires
  objects under the prefix after 7 days (the engine assumes at least a day). `--context-store-ttl-hours`
  governs only the files in this directory; a session only the bucket holds ages there, and drops out
  of the index at the next listing once the bucket no longer lists it. A session that is used has the
  age of its objects restarted (a server-side copy onto itself, no data transferred, at most once a day
  per session), and an object the bucket has lost in the meantime is uploaded again from the directory,
  so something in use does not expire under it. A write is uploaded from a snapshot taken when it was
  written, with its chunk files kept on disk until the upload ends (at most 16 uploads wait, each
  newer write of a session replacing its queued upload); a chunk that is not intact on disk stops the
  upload instead of publishing a session whose data is damaged. A chunk found damaged on disk when a
  session is loaded is fetched again from the bucket.
- An unreachable bucket costs the uploads and the fetches, counted in `ninfer_context_store_remote_*`;
  the directory keeps working. Shutdown waits for queued uploads up to the remote flush budget (two
  minutes) and then interrupts the transfer in progress, so it never waits out a stalled connection.


### Model suspend

With `--model-suspend`, an idle server can give its device memory back to the GPU without exiting,
for instance to let a game or another program use the card for a while, and take it again on the
next request. Every device allocation the model owns (weights, KV cache and recurrent state,
workspace, on every device of a `--devices` split) sits at a fixed address whose physical memory a
suspend releases; the CUDA Graphs, block tables and Host caches stay as they are. A resume maps fresh
memory at the same addresses, restores the weights and copies the retained state back, so
conversations retained before the suspend reuse their prefixes afterwards exactly as before. It
needs CUDA virtual memory management on every device, and it is refused for
Qwen3.8-Flash-Next. It works with `--vision-residency overlay`: the overlay's weight arena and KV
arena release and restore their own chunks like every other region, and the pinned Vision tower stays
in host memory.

A suspend runs between units and only while the server is idle: nothing queued, admitted, paused or
running, no context transaction. Otherwise it answers `409 model_busy`; it never cancels work. A
generation request that arrives while the model is suspended resumes it first, then runs. With
`--no-auto-resume` (or `{"auto_resume": false}` in the suspend request) it fails with `503`
instead and `/health` reports the server unavailable until an explicit resume.

| Endpoint | |
|---|---|
| `POST /v1/models/{id}/suspend` | give the device memory back. Optional body `{"auto_resume": false}` |
| `POST /v1/models/{id}/resume` | map the memory again and restore the weights and state |
| `GET /v1/models/{id}/residency` | `state` (`resident` or `suspended`), `releasable_device_bytes`, `host_snapshot_bytes`, the suspend and resume counts, how long the last of each took and, after a failed resume, `last_error` |

`GET /props` carries `is_sleeping` for the same state. `--sleep-idle-seconds N` suspends the model
by itself after N seconds with no admitted request.

Where the weights come from on a resume is the trade between host memory and wake time
(`--suspend-weights`):

- `artifact` (default) reads them again from the artifact the server loaded, whose files stay open.
  No extra host memory, but a wake costs about what the load did.
- `host` keeps a host copy taken at suspend, as much memory as the weights while suspended, and wakes
  at bus speed.

`--suspend-snapshot` picks where the retained state waits: `pageable` (default) allocates host
memory at suspend and frees it after the resume; `pinned` reserves one page-locked block at startup
for the largest snapshot, which transfers faster and holds that memory for the life of the server.
A failed resume leaves the model suspended with its state intact and the next request retries it.

Measured on an RTX 3090 with the Qwen3.8-27B artifact (4,096 tokens of context, one lane): a suspend
gives back all 15.4 GiB the model's regions held in about 0.15 s, copying 343 MiB of retained state to
host memory. A resume takes 3.3 s with `--suspend-weights artifact` (the file was in the operating
system's cache; a cold read takes as long as a cold load) or 1.9 s with `--suspend-weights host`,
which holds 15 GiB of host memory while suspended. A conversation retained before the suspend still
reuses all 1,324 prompt tokens of its prefix afterwards, and greedy output is identical. The CUDA
context itself and anything allocated outside the model's regions stay resident: a suspended
`ninfer-serve` showed about 230 MiB more in `nvidia-smi` than the idle desktop, against 15.8 GiB
more while resident.


### Several models (router)

Give `ninfer-serve` no artifact and name the models with `--models-dir DIR` (every `.ninfer` in
`DIR`, and each subdirectory holding exactly one, by its file or directory name) or
`--models-preset FILE`, and one server answers for all of them on one address. Models load on demand:
a request names its model in the body's `"model"` (an id or an alias) or a `?model=` query
parameter, and a request that names none goes to the model that last served. Every other option on
the command line applies to each model; a model's section in a preset can change any of them.

```ini
[*]
max-context = 65536
kv-dtype = rk8v4
; a boolean key is a flag; a model's section can drop it with false. A flag that is on by
; default (prefill-a8, cuda-graph, prefix-reuse) is turned off with its no- key, e.g. no-cuda-graph = true
model-suspend = true

[coder]
model = /models/qwen3_8_27b.ninfer
alias = default, qwen
spec = dflash2
draft-tokens = 7

[small]
model = /models/other.ninfer
max-context = 8192
```

`[*]` holds defaults for every model; any other section is a model (`model` or `artifact` is its
path, `alias` a comma-separated list, `load-on-startup` a boolean, every other key a serve option
without its leading dashes). A section named after a model found in `--models-dir` needs no path.
The command line (a `--config` file included) wins over a preset where they repeat an option.
Comments sit on lines of their own (`;` or `#` first). The router's own options (`models-max`, `models-autoload`, `sleep-idle-seconds`,
`unload-idle-seconds`) are command-line only, as are the options the router overrides for every
model (`api-key`, `allowed-host`, `host`, `port`, `model-id`): a preset that sets one is rejected
at startup. A model id may not contain `/`, `?`, `#`, `%` or `\`. With
`--context-store DIR` each model keeps its store in `DIR/<model id>` (and under `<prefix><model id>/`
in S3; characters other than letters, digits, `-` and `_` are written as `%XX`). `--request-log-jsonl`
may not name any model's artifact, n-gram volume or expert state, whether it comes from the command
line or a model's own section; models that name the same file append to one shared log, one record
at a time. The router accepts a request body up to the largest `--max-request-mib` of any model, and
that model's own server then applies its limit.
`--ngram-volume`, `--expert-state` and `--default-expert-state` may not give the same file to two
models, and one model's n-gram volume, expert state, default expert state and artifact must be
different files.

`--models-max N` (default 1, 0 for no limit) bounds how many models are loaded at once. To make room
the least recently used model without a request in flight **sleeps** when it was started with
`--model-suspend` (its device memory is given back, and it wakes in seconds with its retained
conversations) and is **unloaded** otherwise. `--sleep-idle-seconds N` and
`--unload-idle-seconds N` apply the same transitions to idle models. A request for a model that is
not loaded loads or wakes it, unless `--no-models-autoload`. A model whose Engine failed is unloaded
once its requests are gone, so the next request loads it again.

The router answers llama.cpp's model API itself:

| Endpoint | |
|---|---|
| `GET /models`, `GET /v1/models` | every model with its `status.value` (`unloaded`, `loading`, `loaded`, `sleeping`, `unloading`), the arguments it starts with, its aliases and the last failure |
| `POST /models/load` | `{"model": ID}`: load it, or wake it when it sleeps |
| `POST /models/unload` | `{"model": ID}`: unload it once its requests have ended |
| `POST /models/sleep` | `{"model": ID, "hold": false}`: suspend it; with `hold` it stays asleep until loaded again |
| `GET /models/sse` | a stream of `{"model", "status"}` events |
| `POST /v1/models/{id}/suspend`, `/resume`, `GET /v1/models/{id}/residency` | the [model suspend](#model-suspend) endpoints of one model |

`GET /health` is 200 while the router runs. `GET /props`, `/metrics`, `/v1/load` and `/slots`
answer for the model that last served, without waking or loading it (a 503 while it is unloaded).
The router serves `GET /dashboard` itself, and the page reads those status endpoints.
Everything else (chat completions, Responses, Messages and the rest) is forwarded to the model it
names: each model runs the ordinary single-model server on a loopback port of its own, so every
endpoint, request log and metric works per model unchanged. The router checks `--api-key`, runs the
[Host and Origin checks](#host-and-origin-checks) and applies `--cors`; models need none of them.
Streamed responses stream, and a client that disconnects cancels the generation, also while the
model is still loading or working on the prompt. The router rewrites only the `"model"` member of a
body it forwards (to the model's id), so every other value reaches the model as you wrote it.

Each model keeps its own Responses store, and a response ID does not say which model made it. A
request that names no model goes to the model that last served, so `GET` or `DELETE
/v1/responses/{id}`, `.../input_items` and a `previous_response_id` follow-up must name the model
that created the response (`?model=ID` on the request, or `"model"` in the body). Routing these by
the response ID alone is tracked in [#336](https://github.com/ashalliants/ninfer-3090/issues/336).

A model's Responses store survives both sleep and unload (eviction, idle unload, an explicit unload, or
a rebuild after an Engine failure): after the model loads again, `previous_response_id` follow-ups and
`GET`/`DELETE /v1/responses/{id}` still find responses made before. The store lives in memory, so it
is still lost when the router process restarts.

On an RTX 3090 with two copies of the Qwen3.8-27B artifact under `--models-max 1 --model-suspend`: a
cold request loaded its model and answered in 6.0 s (the file was in the operating system's cache),
a request for the other model put the first to sleep and answered in 6.1 s while it loaded, and each
swap after that took 4.0 to 4.1 s with both models staying in the process.

### Metrics

`GET /metrics` serves Prometheus text format 0.0.4 on the same port. Like `/v1/load` it requires
the API key when one is configured, answers `503 model_loading` until warmup completes, and works
independently of `--request-log-jsonl` and `--log-stats-interval-ms`. Engine failure leaves the
endpoint readable with `ninfer_engine_ready 0`. The handler copies published snapshots and formats
them outside the Engine worker; scraping does not reset counters or initiate CUDA work. The one live read is the `ninfer_gpu_*` gauges below, which query NVML (the driver's monitoring library) once per scrape.

```bash
curl http://127.0.0.1:8080/metrics -H 'Authorization: Bearer local-secret'
```

The `llamacpp:` series use llama.cpp's `--metrics` names and meaning, so dashboards and routers
built for llama.cpp work unchanged. They are the Engine's totals since startup and advance per
prefill unit and decode round, during a request rather than at its completion.

| Series | Type | Meaning |
|---|---|---|
| `llamacpp:prompt_tokens_total` | counter | prompt tokens computed by prefill; prefix-cache hits excluded |
| `llamacpp:prompt_seconds_total` | counter | prefill execution time |
| `llamacpp:tokens_predicted_total` | counter | tokens committed by decode rounds |
| `llamacpp:tokens_predicted_seconds_total` | counter | decode execution time |
| `llamacpp:requests_processing` | gauge | admitted requests up to `--max-concurrency` |
| `llamacpp:requests_deferred` | gauge | admitted requests waiting beyond `--max-concurrency` |

Every other series is named `ninfer_*`. Its counters start after startup warmup and reset when the
server restarts.

| Series | Meaning |
|---|---|
| `ninfer_engine_ready`, `ninfer_server_start_time_seconds` | whether the Engine accepts work, and the Unix time the server attached it after warmup |
| `ninfer_model_info`, `ninfer_max_concurrency`, `ninfer_max_context_tokens`, `ninfer_spec_decode_draft_window` | model/backend identity and startup limits |
| `ninfer_requests_running`, `waiting`, `paused`, `prefilling`, `decode_ready`, `replaying`, `materializing` | current Engine gauges; prefill/decode/replay are subsets of resident requests |
| `ninfer_prompt_tokens_total`, `ninfer_prompt_tokens_cached_total` | full input and context-cache-reused tokens, counted once on initial binding |
| `ninfer_prefill_tokens_total`, `ninfer_replayed_tokens_total` | actual initial prefill and separate recovery recomputation |
| `ninfer_generation_tokens_total`, `ninfer_decode_tokens_total` | all committed outputs, or decode/control outputs excluding the first token; include thinking and injected control tokens |
| `ninfer_prefill_seconds_total`, `ninfer_decode_seconds_total` | prefill-unit and decode-round execution time |
| `ninfer_decode_rounds_total`, `ninfer_decode_row_rounds_total` | decode batch executions and the sum of their batch sizes |
| `ninfer_spec_decode_{rounds,draft_tokens,accepted_tokens,fallback_steps}_total` | live native speculative work, including MTP and DFlash/DFlash2 |
| `ninfer_spec_decode_ngram_{rounds,draft_tokens,accepted_tokens}_total` | the n-gram copy rounds among them (`--ngram-draft-tokens`); zero when it is off |
| `ninfer_spec_decode_relaxed_kept_tokens_total` | accepted drafts that [relaxed drafting](#relaxed-drafting) kept and exact verification would have rejected, a subset of the accepted tokens; zero when it is off |
| `ninfer_root_selections_total`, `ninfer_checkpoint_selections_total` | admissions by the context-cache source they started from: root is a miss (full prefill), checkpoint a reuse. Hit rate is `checkpoint / (root + checkpoint)` |
| `ninfer_{preemptions,snapshot_restores,replay_restores}_total` | resource-pressure pauses and recovery routes |
| `ninfer_admission_fit_deferrals_total` | fresh requests queued by `--admission-fit-check` because their whole prompt could not fit in the KV pool beside the resident requests; zero when the flag is off |
| `ninfer_admission_overtakes_total` | fresh requests `--admission-affinity` admitted ahead of older waiters because they resume a prefix held on the GPU; zero when the flag is off |
| `ninfer_waiting_cancelled_requests_total`, `ninfer_waiting_expired_requests_total`, `ninfer_waiting_abandoned_seconds_total` | requests the client cancelled, or the pending timeout expired, before admission, and the total time they had waited (a client that gives up after 60 s shows up here) |
| `ninfer_cancelled_prefills_total`, `ninfer_cancelled_prefill_computed_tokens_total` | requests cancelled while prefilling, and the prompt tokens they had computed |
| `ninfer_paced_prefill_steps_total`, `ninfer_paced_prefill_deferred_tokens_total` | prefill steps cut short by `--prefill-stall-budget-ms` because another lane was decoding, and the prompt tokens they pushed to later steps |
| `ninfer_prefill_backlog_seconds` | gauge: predicted prefill work ahead of a new request (resident remainder plus queued requests uncached); the same figure as `prefill_backlog_seconds` in `/v1/load` |
| `ninfer_prefill_yields_total` | prefills paused so a short request at the head of the queue could be admitted (`--yield-long-prefill-ms`) |
| `ninfer_deferred_prefill_turns_total` | worker cycles that held the prefill step back for `--prefill-decode-share` |
| `ninfer_sliced_prefill_steps_total`, `ninfer_prefill_slice_boundaries_total` | prefill steps run in layer slices (`--prefill-layer-slices`), and the boundaries between their slices where a neighbour's decode round ran |
| `ninfer_engine_recoveries_total` | host-side worker failures the server survived by failing the in-flight requests and clearing the context cache |
| `ninfer_device_kv_{used,capacity}_pages`, `ninfer_device_backend_kv_used_pages`, `ninfer_device_state_{used,capacity}_slots` | physical Main KV, speculative backend KV and StateImage occupancy; retained history also occupies these pools |
| `ninfer_host_context_{used,reserved,capacity,peak}_bytes`, `ninfer_host_state_images`, `ninfer_host_kv_used_bytes` | unified Host backing; reserved bytes, StateImages and KV bytes are breakdowns already included in used bytes |
| `ninfer_host_context_free_extents`, `ninfer_host_context_{geometry,shortage}_failures_total` | Host backing health: separate free runs now (many small ones mean fragmentation), and allocations that failed. A geometry failure had enough free bytes in total but no run large enough; a shortage failure had too few bytes. Non-zero geometry failures mean a demotion or restore was refused for layout, not capacity |
| `ninfer_context_transfer_bytes_total{resource,direction}`, `ninfer_context_transfer_seconds_total` | actual `state`/`main_kv`/`backend_kv` payload transfers, `d2h`, `h2d` or `d2d`, and their time |
| `ninfer_context_pressure_spill_pages_total` | KV pages moved from Device to Host to relieve pressure |
| `ninfer_host_work_seconds_total{phase}`, `ninfer_device_wait_seconds_total` | instrumented worker wall time; device wait is not CUDA kernel time |
| `ninfer_engine_host_seconds_total` | the `host_work` phases summed: Engine host-active seconds, device wait excluded. Its rate against token throughput exposes a host-bound Engine |
| `ninfer_token_intervals_total`, `ninfer_token_interval_seconds_total` | gaps between output tokens of finished requests and their first-to-last-token wall time; the ratio of their rates is the mean inter-token time |
| `ninfer_last_request_inter_token_seconds`, `ninfer_last_request_decode_host_seconds` | gauges: mean inter-token time and decode host time of the last finished request with two or more tokens (absent before the first). A healthy Engine stays in milliseconds; one that stalls after every first token reports seconds while `/health` stays 200, which a supervisor can restart on |
| `ninfer_constraint_requests_total{outcome}`, `ninfer_constraint_cache_total{result}` | settled constrained requests by completion state, and compilation-cache access |
| `ninfer_constraint_{prepare,mask,matcher}_seconds_total`, `ninfer_constraint_mask_{positions,upload_bytes}_total` | [constraint](#output-constraints) work aggregated at request settlement, including truncated/cancelled results |
| `ninfer_constraint_draft_wait_seconds_total` | live draft-ready wait counted once per batch; a subset of device wait |
| `ninfer_requests_total{outcome}`, `ninfer_response_failures_total` | generation attempts entering preparation by outcome (`completed`, `cancelled`, `failed`, `rejected`), and response failures after settlement; protocol/model validation failures and token-count requests are excluded |
| `ninfer_time_to_first_token_seconds` | histogram updated once at the first committed token, including preparation, queueing and binding |
| `ninfer_request_duration_seconds`, `ninfer_request_queue_seconds` | histograms for settled generation outcomes, including cancellation; exceptional failures have separate counts |
| `ninfer_gpu_utilization_percent`, `ninfer_gpu_memory_utilization_percent` | gauges: share of NVML's last sample period in which any kernel ran, and in which device memory was read or written. Measured on the first device (`--device`, or the first of `--devices`) |
| `ninfer_gpu_memory_used_bytes`, `ninfer_gpu_memory_total_bytes` | gauges: device memory in use by every process on the card (not only this server), and installed |
| `ninfer_gpu_temperature_celsius`, `ninfer_gpu_power_watts` | gauges: GPU core temperature and board power draw |
| `ninfer_gpu_power_limit_watts` | gauge: the board power limit the driver enforces now, after every limiter (`nvidia-smi -pl` included). A capped card is slower, so read throughput against it. All seven `ninfer_gpu_*` gauges are omitted, not reported as 0, when NVML or the device cannot be opened (no driver library, a container without it); a single reading NVML cannot give is omitted on its own |
| `ninfer_gpu_energy_joules_total` | counter: board energy used since the server attached to the GPU, from NVML's lifetime energy counter. The rate over any window is the mean power over it (`rate(ninfer_gpu_energy_joules_total[5m])` is watts), which the instantaneous `ninfer_gpu_power_watts` cannot give. It starts at 0 at attachment, not at driver load, and keeps counting through an NVML counter reset. Omitted, like the other `ninfer_gpu_*` series, when NVML cannot supply it (older drivers, a card without an energy counter) |
| `ninfer_context_store_images`, `ninfer_context_store_used_bytes` | gauges: sessions and bytes held by the [context store](#context-store) (zero when it is off) |
| `ninfer_context_store_writes_total`, `_write_failures_total`, `_dropped_total` | sessions written, writes that failed, sessions not queued because the write queue was full |
| `ninfer_context_store_bytes_written_total`, `_bytes_stored_total`, `_bytes_reused_total` | new chunk bytes written (raw size), what they took on disk after [compression](#context-store), and chunk bytes a write found already stored |
| `ninfer_context_store_peer_images_total`, `_peer_bytes_linked_total`, `_peer_skipped_total`, `_peer_link_failures_total` | with `--context-store-peer`: sessions imported from peers, chunk bytes hard-linked (shared on disk, not copied), sessions not imported (damaged, expired, over budget, or gone mid-import) and chunks that could not be linked |
| `ninfer_context_store_evicted_total`, `_corrupt_total` | sessions removed for space, age or supersession, and because they could not be read back intact |
| `ninfer_context_store_restored_sessions`, `_restored_bytes`, `_restore_seconds` | gauges: what start-up restored into the cache, and how long it took |
| `ninfer_context_store_hydrations_total`, `_hydrated_tokens_total`, `_hydration_failures_total`, `_hydration_seconds_total` | stored sessions read back for a request, the prompt tokens that saved, failures (the request was prefilled, or its admission did not use what was read) and the time requests waited for the reads, failed ones included |
| `ninfer_context_store_shared_writes_total`, `_shared_export_seconds_total` | public shared prefixes written to the store, and the Engine worker time spent copying them out |
| `ninfer_context_store_eviction_writes_total`, `_eviction_export_seconds_total` | conversations queued for the store as the cache evicted them, and the Engine worker time spent copying them out |
| `ninfer_context_store_read_bytes_per_second` | gauge: the store read rate (read, verification and import into the Host tier) the cost model prices hydrations with, learned from the reads so far; 1 GB/s until the first read of at least 64 MiB |
| `ninfer_context_store_remote_images` | gauge: with a bucket, sessions it holds that the directory does not hold in full |
| `ninfer_context_store_remote_uploads_total`, `_remote_upload_bytes_total`, `_remote_upload_failures_total` | objects and bytes uploaded to the bucket, and uploads that failed |
| `ninfer_context_store_remote_downloads_total`, `_remote_download_bytes_total`, `_remote_download_failures_total` | chunks and bytes fetched from the bucket, and listings or fetches that failed or returned damaged data |
| `ninfer_context_store_remote_manifest_downloads_total`, `_remote_manifest_download_bytes_total` | session manifests a bucket listing downloaded to register sessions other engines wrote (not in the chunk counters above) |
| `ninfer_context_store_lookups_total`, `_lookup_seconds_total` | bucket listings refreshed for a request that found nothing stored, and the time requests waited for them |
| `ninfer_context_store_remote_fetch_bytes_per_second` | the rate the store assumes for fetching a session's missing chunks (downloaded, verified and written to the directory together: the wait a fetch adds), learned from fetches; a session whose fetch would not fit in a request's wait is not chosen |
| `ninfer_context_store_remote_retired_total`, `_remote_uploads_skipped_total` | earlier turns deleted from the bucket once a later one was uploaded, and queued uploads a later write made unnecessary |

Histograms expose `_bucket`, `_sum` and `_count`. Metrics have bounded labels; they do not retain
request IDs or request text. Rates are calculated by the consumer, for example:

```promql
rate(ninfer_generation_tokens_total[1m])

rate(ninfer_spec_decode_accepted_tokens_total[1m])
/ rate(ninfer_spec_decode_draft_tokens_total[1m])
```

Detailed per-request records remain available through the
[structured request log](#structured-request-log). Earlier releases of this fork named these series
`ninfer:*`; colons are reserved for Prometheus recording rules, so they are now `ninfer_*`, and the
colon names are no longer emitted.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history;
- string content and ordered text/refusal parts; adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn;
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension using HTTP(S) or data URIs; image detail is omitted or `auto`;
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; zero performs prompt
  processing without generation, and omitting both applies the
  [default output limit](#default-output-limit);
- `temperature`, `top_p`, presence/frequency penalties, and signed integer `seed`;
- the compatible `top_k` (`0..20`) and `min_p` (`0..1`) sampler extensions;
- the NInfer `relaxed_drafting` extension (a number in `[0,1]`, `true` for 0.5, `false` for
  exact), which CHANGES THE OUTPUT DISTRIBUTION; see [Relaxed drafting](#relaxed-drafting);
- up to four non-empty stop strings, applied to both reasoning and answer output;
- the `ignore_eos` benchmarking extension shared with vLLM, SGLang and llama.cpp: a boolean,
  default `false`, that drops the checkpoint's own stop tokens so generation runs to the output
  budget (`finish_reason:"length"`) unless a caller-supplied stop string or context capacity ends
  it first. Caller stop strings still apply; any other value type is rejected with 400. Output past
  the model's natural end is not meaningful text. `/v1/responses` and `/v1/messages` do not
  honor it;
- `n:1` and text-only `modalities`;
- `response_format` of type `text`, `json_object`, or `json_schema`, and the `structured_outputs`
  extension (grammar, choice or regex); see [Output constraints](#output-constraints);
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- function tools, optional `strict:true` argument schemas, `tool_choice` `auto`/`none`/`required`,
  named selection, `allowed_tools`, and `parallel_tool_calls`; assistant tool-call history,
  tool-result messages, and legacy function-call history;
- free-form [`custom` tools](#chat-custom-tools) in definitions, named selection,
  `allowed_tools` and assistant history;
- the top-level `reasoning_effort` field;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- the `graft` extension selecting a [prompt graft](#prompt-grafts);
- the `thinking_budget` extension, a positive per-request [thinking cap](#openai-chat-completions);
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes nonzero `logit_bias`, requested log probabilities,
audio/file input or audio output, explicit low/high image detail, web search,
moderation, low/high verbosity, stored Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.

### Output constraints

JSON mode and JSON Schema use the standard protocol fields:

| Endpoint | Field |
|---|---|
| Chat Completions | `response_format: {"type":"json_object"}` or `{"type":"json_schema","json_schema":{"name":"answer","schema":{...},"strict":true}}` |
| Responses | `text.format: {"type":"json_object"}` or `{"type":"json_schema","name":"answer","schema":{...},"strict":true}` |
| Anthropic Messages | `output_config.format: {"type":"json_schema","schema":{...}}` |

`json_object` requires an object root. Schema mode follows the supplied root type and enforces the
[supported assertions](maintainer/constrained-decoding.md#42-json-与-schema-的执行合同), including when
`strict` is omitted or false. Unsupported assertions return HTTP 400 before generation. OpenAI
errors distinguish `invalid_json_schema`, `unsupported_json_schema` and `unsatisfiable_json_schema`;
`param` identifies the request field followed by the schema JSON Pointer. Anthropic uses its
`invalid_request_error` envelope with the schema location in the message. Responses echoes the
selected `text.format` in aggregate responses and SSE response objects.

JSON output may use any layout, compact or indented (whitespace runs are capped at 32
characters), and follows declared property order. State the desired content in the prompt; the
schema is not inserted into it. Only one output constraint may be supplied. `json_object` also
accepts `"schema":{"type":"object"}`, which llama.cpp-style clients send; any other schema belongs
in `json_schema`.

Schemas support positional arrays (`prefixItems` plus tail `items`) and inclusive/exclusive
`number` ranges. Bounded numbers use exact int64 integers or finite binary64-compatible decimal
and scientific notation with up to 17 significant digits. Bounds must retain their value when the
schema is parsed; numbers requiring greater precision receive `unsupported_json_schema`.
These capabilities also apply to strict tool parameters.

GBNF, choice and regex are available through the NInfer extension `structured_outputs`
on Chat Completions, Responses and Anthropic Messages. Supply exactly one member:

```json
{"structured_outputs": {"grammar": "root ::= \"yes\" | \"no\""}}
```

```json
{"structured_outputs": {"choice": ["positive", "neutral", "negative"]}}
```

```json
{"structured_outputs": {"regex": "(BUG|TASK)-[0-9]{4}"}}
```

Choice returns one literal string, preserving case and whitespace. The list must be nonempty;
duplicate entries have no extra weight, and an empty-string entry permits empty content.
Regex matches the complete content. It supports character classes, groups, alternatives and
repetition; `.` excludes line terminators, `\d`/`\w` use ASCII ranges, and `\s` includes Unicode
whitespace. Empty regex permits only empty content. Anchors are supported at the ends of top-level
alternatives. Lookaround, backreferences, word boundaries, Unicode properties, flags and unknown
escapes return HTTP 400. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice).
Invalid choices and regexes use `invalid_choice` and `invalid_regex`, with the request field in `param`.

These constraints apply to answer content; thinking is separate. GBNF supports recursive rules,
Unicode and repetition. All modes support streaming and all speculative backends. For assistant
continuation, the grammar covers the existing assistant content plus the generated suffix. Completion uses the model's EOS tokens;
output limits and cancellation can produce an incomplete answer. JSON modes can be combined with
active tools; GBNF, choice and regex require no active tools or `tool_choice:"none"`.
Output constraints reject custom stops. OpenAI errors use
`invalid_grammar` for invalid grammars and `constraint_dead_end` for a reachable prefix without a
legal next token. Anthropic reports these through its `invalid_request_error` envelope.
The `grammar` and `guided_*` aliases are not accepted.

Constrained responses include a NInfer `constraint` observation. `branch` is `undecided`, `content`,
or `tools`; `complete` means the committed language can end, and `terminated` means it accepted EOS.
A complete JSON value can therefore have `complete:true`, `terminated:false` and a length finish
reason. The observation also includes `cache` (`hit`, `built`, `waited`), `mask_positions`,
`mask_upload_bytes`, and `timings_seconds` for preparation, CPU mask work and matcher work.
These times are parts of existing request time and can overlap GPU execution.
Streaming sends the observation once: the Chat finish/usage chunk, the Responses terminal response
object, or Anthropic `message_delta`. Unconstrained responses omit it.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

A string `name` on any message is accepted and ignored, as the optional OpenAI participant label
that agent frameworks use to tag injected messages and that some clients mirror onto tool results.
It does not participate in tool identity, prompt rendering, or output, so distinct names do not
create distinct participants. Non-string values are malformed.

Function names replayed in message history (`tool_calls`, `function_call`, `function` messages)
must be non-empty strings but are not restricted to the definition charset, so history from other
clients or models round-trips unchanged. Tool definitions still require `[A-Za-z0-9_-]{1,64}`.
`allowed_tools` entries use the Chat shape `{"type":"function","function":{"name":...}}`; the flat
`{"type":"function","name":...}` form is also accepted.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1`, and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors.

The request `model` must equal the public model ID: the artifact `metadata.name` by default
(falling back to its architecture name when absent), or the explicit `--model-id` override.
Reasoning is returned separately as `reasoning_content`; answer text remains in `content`.
The thinking ends at a `</think>` followed by a line break (the model writes `\n</think>\n\n`) or by
the end of the turn. A `</think>` that thinking quotes while it discusses the tag, followed by a
space, a quote, a backtick or other text on the same line, stays reasoning, so the rest of the
thinking is not published as the answer. Constrained output (tools under `basic`, JSON formats)
still requires the exact `\n</think>\n\n` framing.

For non-strict tools, a direct top-level tool-parameter
`type`, or an `anyOf`/`oneOf` composed entirely of explicit primitive types, guides conversion of
Qwen's untyped parameter text. It does not decide whether structurally complete markup is a tool
call. String-admitting values remain strings, including the empty string. An empty block for a
declared non-string parameter is omitted. Admitted JSON values retain their JSON type;
case-insensitive boolean text is normalized to `true` or `false`. A nonempty schema mismatch remains
a structured call: valid JSON retains its represented type and other text becomes a JSON string so
the tool consumer can report the validation error and continue the agent loop. Schemas without a
supported explicit type retain untyped inference. NInfer does not apply defaults, enforce required
properties, or perform recursive JSON Schema validation on this route.

On the unconstrained route, the parser also accepts the XML call forms agent harnesses such as
Claude Code prompt models with: a `<function_calls>` wrapper, bare `<function ...>` and
`<invoke ...>` calls (closed by the matching `</function>` or `</invoke>`), `name="..."` or
`name='...'` attributes in place of `=name`, and `<param ...>...</param>` parameters. Opening and
closing tags must match. Repeated parameter names keep the last value at the first key's position,
as on the constrained route; `duplicate_parameters_repaired` in the
[structured request log](#structured-request-log) counts them.

String parameters preserve function/tool-call markers and balanced nested
`<parameter=...>...</parameter>` text as value bytes. The Qwen wire format has no delimiter escape,
so a standalone `</parameter>` ends a value only when whitespace and then another parameter, the
function's closer, or the end of the output follow it; any other one is value text, such as a shell
command that echoes the markup. An unmatched nested parameter opener, or a quoted closer that is
followed by the next token, cannot be represented unambiguously and makes that tool-call region
ordinary content. Later content is still examined: the first tool-call region (any accepted marker
form, or a later `<tool_call>` after one that failed) that parses to the end of the output becomes
the structured turn, and any quoted markup before it stays ordinary content. A marker inside a
Markdown code fence (```` ``` ```` or `~~~`) or inline code of the answer is a quoted example and
stays content even when it is a complete call; only the text before the marker decides, so streamed
and complete responses agree. A marker after a closed fence, or after a sentence on the same line,
still opens a call. Inline code is tracked per paragraph (a blank line or a fence ends one), so an
unpaired backtick earlier in the same paragraph also makes a later marker text.

A model sometimes writes its tool calls inside its thinking and ends the turn with its stop token
without ever closing the thinking, so no call reaches the answer. On the unconstrained route such a
call is **executed**: when, from a marker (found by the same rules as in the answer) to the end of
that thinking, there is nothing but complete calls to declared tools, those calls become the
structured turn, and the operational log notes the recovery at Info severity. This means a call the
model wrote while still thinking is returned for execution when the model ended its own turn right
after complete, declared calls. The thinking stays reasoning, including the calls' text, and none
of it is published as content. Nothing is recovered when the thinking was closed, when the turn
published any content, when it ended at the output limit, context capacity, a stop string or a
cancellation, or when prose, a quoted marker or a cut call follows the last call. Constrained tool
output cannot end inside thinking: its grammar admits the stop token only after the canonical
thinking close, so this recovery does not apply there.

#### Tolerant tool calls

By default a tool-call region the parser cannot read exactly is returned as ordinary content, so an
agent client shows raw `<tool_call>` text where a tool should have run. `--tolerant-tool-calls`
(off by default) repairs the common ways a model breaks that markup and returns the calls
structurally instead. It applies only to free tool output, that is requests sent with
`tool_constraints:"auto"` that fall back to free generation as described under
[Tool constraints](#tool-constraints). Constrained output, the default, already admits only exact
calls and never consults the flag.

With the flag, these calls are kept whatever ended the turn:

- complete calls followed by prose or by a malformed or cut call: the calls are kept and what follows
  them is dropped. Prose that itself contains another call marker is not dropped: the calls before it
  may be an example the answer quotes, so the later region is tried instead;
- complete calls inside a `<function_calls>` wrapper that never closes;
- a call whose opener the model mangled inside `<tool_call>`: a dropped or doubled `<`, a leaked
  `<|im_start|>`, a dropped `function` keyword (`<=name>`), or a missing `>` after the function name
  when the next tag follows on the next line;
- a complete call to a tool name the request did not declare. It is returned for the client to
  reject, since leaking it as text would hide the call; strict parsing returns it as text.

A final call whose parameters all closed but whose `</function>` or `</tool_call>` is missing is
kept only when the model ended the turn itself with its stop token. If the turn ended at the output
limit, context capacity, a stop string or a cancellation, the call may have been cut between
parameters, so only the calls before it are kept, as on the constrained route.

The flag **never** keeps a call whose parameter value was cut off before its closing tag: a cut path,
command or patch would run as something the model did not write. Such a call is dropped (or, when
it is the only one, the output stays text). For the same reason it drops nothing that holds a
`</parameter>`, `</param>`, `</function>` or `</invoke>` closer before the next call marker: a
value that quotes a closer followed by the next token (`echo '</parameter>` then `</function>'`)
would have been split there, so the region stays text as under strict parsing. The flag also
never applies to calls recovered from unclosed thinking, which must parse exactly. A repaired turn
reports `tolerant_recovered` in the [structured request log](#structured-request-log) and logs one
Info line.

### Tool constraints

The three protocols share one constrained tool implementation:

| Choice | Generated calls |
|---|---|
| `auto` | Text or calls; zero to many |
| `none` | No tool calls; declarations remain in the prompt |
| OpenAI `required` / Anthropic `any` | One or more calls |
| OpenAI named function or custom tool | Exactly one call to that tool |
| Anthropic named `tool` | One or more calls to that tool |
| OpenAI `parallel_tool_calls:false` / Anthropic `disable_parallel_tool_use:true` | At most one call; exactly one when a call is required |

OpenAI `allowed_tools` supports `auto` and `required`. Selection changes generation permissions,
while all declarations retain their original order in the prompt. Requests with tools enable
**basic structural constraints by default**, including ordinary `auto` calls without `strict`.
The model can answer normally or start a tool call; a call must use the model's tool framing and a
declared function name.

Non-strict parameter names and order remain open. Their schema supplies the existing value
normalization hints; it is not compiled as a strict constraint. Open objects, root unions, and
unsupported schema assertions therefore remain usable. Repeated parameter names use the last
value, retaining the first key position; the published JSON object contains each key once.
`strict:true` additionally enforces the parameter contract below.

Top-level `tool_constraints:"auto"` opts into request-driven constraints: ordinary non-strict
`tool_choice:"auto"` then uses free generation. Strict tools, selection/count restrictions, and
`tool_choice:"none"` still enforce their requirements. `tool_constraints:"basic"` is the default.

With JSON object/schema output, `auto` permits either a JSON answer or a complete tool-call sequence.
Required/named choices permit calls for that turn; after supplying the tool result, use `auto` for
the final JSON answer. `none` permits only JSON. The JSON schema is validated on every turn.
This combination enforces tool framing even with `tool_constraints:"auto"`; `strict` continues to
control argument-value validation. Tool markers inside JSON strings remain ordinary string data.

A function's `strict:true` also constrains its argument values against its schema. Its parameter
root must reduce to a `type:"object"` schema with `additionalProperties:false`, including supported
`allOf` and local-reference combinations.
Properties are emitted in declaration order; optional properties may be omitted. Root
const/enum/unions are not supported. Values use the supported JSON Schema subset described above.
Top-level pure string parameters use raw text and preserve whitespace. Other values use JSON;
a top-level string/non-string union (such as string/null) is rejected because the Qwen parameter
format cannot distinguish those branches. Such unions inside JSON objects or arrays are supported.
Raw values cannot contain the delimiter `\n</parameter>`; unsatisfiable required values are rejected.
Integer arguments use signed 64-bit values; number arguments use finite binary64-compatible
representations. Unsupported schemas fail with HTTP 400 before generation.

For `auto` without JSON output, text can precede the first call. Required/named choices start directly with calls
(after thinking, if enabled). Once a constrained call starts, the suffix consists of complete calls
and model EOS. The grammar starts a call at a `<tool_call>` wherever it appears, including inside
Markdown code, so a model that shows the call format in a code fence (```` ``` ```` or `~~~`) or
inline code is made to complete a real call there. Under `auto` (calling optional, including
`allowed_tools` with `mode:"auto"` and `parallel_tool_calls:false`), a call region that opened
inside Markdown code is therefore returned as ordinary content, byte for byte, with no tool calls
as a plain text answer (`stop`, Anthropic `end_turn`, Responses `completed`), so an agent client never executes an example; the
request log records `fallback_reason: quoted_in_code`. The same Markdown rules as on the free route
decide (only the text before `<tool_call>` counts). Required and named choices must call, so their
calls are always structured. Active tool constraints require model EOS, so they cannot coexist with custom stop
strings or `ignore_eos`. Basic constraints are only a default: a request that sends stop strings
or `ignore_eos` with ordinary non-strict `auto` tools and no `tool_constraints` field falls back to
free tool-call generation instead of being refused, because many OpenAI-compatible agent clients
send stop strings with every request. A request that asks for constraints explicitly
(`tool_constraints:"basic"`, a required or named `tool_choice`, `parallel_tool_calls:false` or a
strict tool) together with custom stops is still rejected with HTTP 400 on `stop` (or
`ignore_eos`).
Token limits and cancellation can still stop generation: only completed calls are published.
A later call truncated by the token limit keeps `length`/`max_tokens`/Responses `incomplete` as the
terminal status. Streaming publishes each completed call in the terminal event sequence;
arguments are not streamed incrementally.
Assistant continuation may finish a partial call; a prefix containing a completed call is rejected.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.
Generated token segmentation can also differ from re-encoding the same text, limiting prefix reuse.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Clients send a wider effort vocabulary than the maintained Qwen templates accept (`low`, `medium`,
`xhigh`). This fork collapses `minimal` onto `low` and `high`/`max` onto `xhigh` before rendering,
so OpenAI and Claude Code requests that send `high` do not fail inside the template.

Control-token spellings quoted in message content, tool data or ordinary template kwargs are
encoded as text. Media placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. A request overrides this default with a positive
integer: `thinking_budget` on Chat Completions and Responses (a NInfer extension, accepted at top
level), or `thinking:{"type":"enabled","budget_tokens":N}` on Anthropic Messages (a value of at
least 1024 and below `max_tokens`, per that protocol). A request that does not think receives no
cap, whatever it sends.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

At the cap boundary, Engine first honors a natural `</think>`, stop condition, cancellation, or
total output/context limit. If thinking remains open, it commits Qwen's canonical early-close
guidance and close marker to the same model sequence without sampling, streams the guidance as a
reasoning delta, and continues normal content or tool-call generation. Inserted tokens count in
completion usage and the request's `max_tokens`/`max_output_tokens` budget.

Near the end of the output window the early-close guidance may not fit. Let B be the requested
thinking budget, C the output the request can still produce (the smaller of its output limit and
the context left after the prompt, counting the final writable position), and R the number of tokens
that early close needs: the tokenizer-derived guidance and close marker plus one post-close model
token. R depends on the model's tokenizer and is not a fixed number. A request is never rejected
because C falls in B < C < B + R; Engine resolves it as follows:

- C <= B, or C - B >= R: the budget is B and early close is available, as above.
- C > R (and C - B < R): the effective budget is C - R, which is always below B. Early-close
  guidance and at least one post-close model token still fit.
- C <= R (and C > B): the budget cannot be enforced, so thinking runs to the output limit with no
  guidance inserted. The response can end inside the reasoning with empty content.

Logs report the requested budget, and the effective budget when it differs. A request that cannot
enforce its budget reports the requested value only. The server does not promise that the model will
emit nonempty content or a tool call after the marker. `model_thinking_tokens` in the request log
counts the model's thinking tokens for every thinking request, with or without a budget.

`--reasoning-loop off|stop|conclude` (default `off`) guards thinking-enabled requests against a
model that keeps rewriting the same passages in its thinking, which token penalties do not catch
because every pass varies a little.

- **Check.** Every 512 model-origin thinking tokens, at the end of a decode round, the thinking is
  split into words (letters, digits and non-ASCII bytes; ASCII case folded) and punctuation marks.
  The guard fires when at least 25 % of the last 2,000 words lie inside a 12-word passage that
  occurs at least three times in the last 30,000 words. A passage repeated twice (a recap) does not
  count. The measure and thresholds are those of Strata's `reasoning_loop_recovery` (measured on
  other models; NInfer has not re-tuned them).
- **`stop`** ends the reply there with `finish_reason` `length` (Anthropic `max_tokens`).
- **`conclude`** commits the same control span a reached thinking budget does (Qwen's early-close
  guidance plus `</think>`), so the model answers from the thinking it has; no prompt is read again.
  When the remaining output budget cannot hold the span and one more token, or the request's
  thinking budget could not be enforced (early close unavailable, above), it stops instead.
- It applies to constrained requests too (tool constraints, `response_format`): the control span
  goes through the same grammar-checked path as a thinking budget, so a constrained request can
  still answer or call a tool after `conclude`. With `stop`, a constrained request ends as an
  interrupted one (completed calls only).
- The console line and the request log (`reasoning_loop_detected`,
  `reasoning_loop_thinking_tokens`, `reasoning_loop_coverage`) record each firing; `server_start`
  records the mode as `reasoning_loop`. It can mistake a long, deliberately repeated checklist for a
  loop; keep it off where the thinking legitimately repeats whole passages.

For Chat Completions, `reasoning_effort: "none"` requests disabled thinking. The other standard
values (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`) reach the template on its three rungs:
`minimal` runs as `low`, `high` and `max` as `xhigh`, so clients such as Claude Code that send
`high` work against the bundled Qwen templates.

`--reasoning-effort minimal|low|medium|high|xhigh|max` sets the effort of every thinking-enabled
request that states none on any endpoint, collapsed onto the same rungs. A request effort
(`reasoning_effort`, Responses `reasoning.effort`, Anthropic `output_config.effort`, or
`chat_template_kwargs.reasoning_effort`) overrides it. The default never enables thinking: requests
that run without thinking, through `--no-thinking` or their own options, receive no effort.
Conflicting explicit `enable_thinking` and effort values return `conflicting_template_option`.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unspecified
thinking, effort and preservation options use the template's defaults.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

### Chat custom tools

Chat Completions accepts free-form `custom` tools in the OpenAI Chat shape:

```json
{
  "type": "custom",
  "custom": {
    "name": "apply_patch",
    "description": "Edit files with a patch.",
    "format": {"type": "grammar", "grammar": {"syntax": "lark", "definition": "start: ..."}}
  }
}
```

`format` is omitted, `{"type":"text"}`, or a `grammar` with `syntax` `lark` or `regex` and a string
`definition`. A custom tool is lowered exactly as on [Responses](#custom-tools): the model sees a
strict function under the tool's name with one required string parameter, `input`, whose
description carries the grammar. The grammar is **advisory** (shown, not enforced); the strict
lowering is enforced, so `input` is the call's only argument and its text is returned byte for byte,
except that it cannot contain a line break directly followed by `</parameter>`. Like any strict tool,
a request declaring a custom tool is constrained even with `tool_constraints:"auto"`, so it cannot
use custom `stop` strings or `ignore_eos` (HTTP 400 on `stop`/`ignore_eos`).

A generated call is answered in the OpenAI Chat custom tool call shape, in `message.tool_calls`:

```json
{"id": "call_...", "type": "custom", "custom": {"name": "apply_patch", "input": "*** Begin Patch
..."}}
```

Streaming sends the same object with its `index` in one `delta.tool_calls` entry
(`{"index":0,"id":"call_...","type":"custom","custom":{"name":"apply_patch","input":"..."}}`); as for
function calls, the complete input is sent once generation finishes, so the streamed `input`
pieces concatenate to the aggregate value. `finish_reason` is `tool_calls`.

Custom tools may be forced with `tool_choice: {"type":"custom","custom":{"name":...}}` and listed in
`allowed_tools` as `{"type":"custom","custom":{"name":...}}` (the flat `{"type":"custom","name":...}`
is also accepted there, as it is for function entries). An assistant history entry
`{"id":...,"type":"custom","custom":{"name":...,"input":"..."}}` is replayed as the same Engine call
(`{"input": "..."}` arguments); its result is an ordinary `tool` message. Referring to a declared
custom tool as a function, or the reverse, fails with HTTP 400: `duplicate_tool_name` when both are
declared in `tools`, `invalid_tool_choice` in `tool_choice`, and `invalid_tool_history` in
`messages`. Calls of undeclared tools in history are accepted, as for functions.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The current Vision runtime also
has a 32,768 merged-token envelope (131,072 raw patches); the effective Vision limit is therefore
`min(--max-context, 32768)`. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI prompt caching

Chat Completions and Responses translate OpenAI cache hints into optional shared-prefix write
candidates:

- omitted `prompt_cache_options` creates a default implicit candidate at the end of the latest
  cacheable content part;
- `mode:"implicit"` requests the same automatic candidate explicitly;
- `mode:"explicit"` disables that implicit write for the request;
- `prompt_cache_breakpoint:{"mode":"explicit"}` on supported content creates an explicit
  candidate.

The automatic candidate precedes the message closing tokens, allowing reuse when the same message
body grows and its previous tokens remain an exact prefix. An explicit marker keeps its requested
location. Responses applies this policy after expanding stored history.

One request carries at most four writes. Explicit markers take precedence: four explicit candidates
leave no extra slot for an automatic candidate. If more explicit markers appear in the history, the
latest four remain write candidates. Exact reads of already-published prefixes do not require the
request to repeat a marker.

These fields are optimization hints. A legal boundary that cannot be represented as an exact
rendered-token frontier is ignored without changing prompt content. `prompt_cache_key` is not a
prefix identity, but on Chat Completions and Responses a non-empty key (at most 256 bytes) tells the
Engine the conversation will be extended, and it keeps a private long-anchor checkpoint at the end of
the prompt, so an agent loop with no new user turn between requests resumes from its previous request.
Valid TTL/retention values are accepted, but NInfer does not
promise their wall-clock residency; physical retention follows the resource scheduler.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with the [default output limit](#default-output-limit) but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; other standard effort values pass to the selected template |
| `reasoning.summary`, `reasoning.generate_summary` | `auto`, `concise` or `detailed` (other values are rejected, and the two must agree when both are sent); accepted as a hint only. NInfer produces no reasoning summaries, so the response reports `reasoning.summary: null` and an empty `summary` on reasoning Items. `reasoning.context` and `reasoning.mode` are rejected with `reasoning_option_not_supported` |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `graft` | NInfer extension: name of a [prompt graft](#prompt-grafts), or `null`; also accepted by input token count |
| `thinking_budget` | NInfer extension: positive per-request [thinking cap](#openai-chat-completions), or `null`; rejected by input token count, which does not generate |
| `relaxed_drafting` | NInfer extension: this request's [relaxed drafting](#relaxed-drafting) threshold, a number in `[0,1]`, `true` (0.5), `false` (exact) or `null` (the server default); CHANGES THE OUTPUT DISTRIBUTION |
| `text.format` | `text` (default), `json_object`, or `json_schema`; see [Output constraints](#output-constraints) |
| `tools` | direct function and free-form `custom` definitions, or namespace groups containing them; see below |
| `tool_choice` | `auto`, `none`, `required`, a named function or `custom` tool, or `allowed_tools` of function and custom entries with mode `auto`/`required`; namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` enforces at most one call |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, and explicit breakpoints follow [OpenAI prompt caching](#openai-prompt-caching); `safety_identifier` and `user` are accepted as client hints |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with `reasoning_text` content (`content` itself may be omitted); summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |
| `custom_tool_call` | completed assistant call of a [custom tool](#custom-tools) with optional `id` and namespace, plus required `call_id`, `name`, and string `input` |
| `custom_tool_call_output` | as `function_call_output`, for a custom tool's call |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call`/`custom_tool_call`. Multiple message
Items append their content parts, multiple calls retain declaration order, and a reasoning-only turn
is retained. A user, system, developer, `function_call_output`, or `custom_tool_call_output` Item
ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Write selection follows
[OpenAI prompt caching](#openai-prompt-caching); boundaries affect reuse opportunities, not prompt
identity or output semantics. String message status/phase metadata is accepted but has no Qwen
prompt representation.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning metadata without raw
reasoning text, partial tool Items, and other Item/content types are not supported. HTTP media URLs
stored in a response chain are fetched again when that chain is continued; use data URIs when the
historical media bytes must be immutable.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions'
nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": false
  },
  "strict": true
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

NInfer gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Selection and strict argument enforcement follow the common tool contract above.

### Custom tools

Free-form `custom` tools take raw text instead of JSON arguments. Codex declares `apply_patch` this
way:

```json
{
  "type": "custom",
  "name": "apply_patch",
  "description": "Use the `apply_patch` tool to edit files.",
  "format": {"type": "grammar", "syntax": "lark", "definition": "start: begin_patch hunk+ end_patch\n..."}
}
```

The model sees a strict function under the tool's own name with one required string parameter,
`input`; the function keeps the tool's description, and the parameter's description says the input
is passed exactly as written and, for a `grammar` format, includes its `syntax` and `definition`.
The declared format is **advisory**: it is shown to the model, but neither a lark nor a regex
grammar is enforced on the generated text. What is enforced is the strict lowering: tool framing is
constrained, `input` is the call's only and required argument, and its text is returned byte for
byte (leading spaces, blank lines and all), except that it cannot contain a line break directly
followed by `</parameter>`, which is the Qwen tool format's value delimiter. Like any strict tool,
a request declaring a custom tool is constrained even with `tool_constraints:"auto"`, so it cannot
use custom `stop` strings.

A generated call returns as a `custom_tool_call` Item (`ctc_...`) with a `call_id` (`call_...`),
`name` (and `namespace` when declared in one) and the raw `input`. Streaming emits
`response.output_item.added` (with empty `input`), one `response.custom_tool_call_input.delta`
carrying the input, `response.custom_tool_call_input.done` with the complete input, then
`response.output_item.done`; as for function calls, these events are sent once generation finishes.
The client returns its result as a `custom_tool_call_output` Item. Custom tools may be named by
`tool_choice: {"type":"custom","name":...}` and listed as `custom` entries in `allowed_tools`.
Referring to a custom tool as a function, or the reverse, fails (`duplicate_tool_name` in `tools`,
`invalid_tool_choice` in `tool_choice`, `invalid_tool_history` in `input`). Chat Completions
accepts custom tools in its own wire shape; see [Chat custom tools](#chat-custom-tools).

Hosted tools, remote MCP tools, deferred loading, output schemas, and caller restrictions that
exclude direct invocation remain unsupported.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` or [`custom_tool_call`](#custom-tools) Items.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`, whose `error.code` is one of the wire enum
values (`server_error` unless the failure already uses one) with the detail in `error.message`;
validation and preparation errors remain normal HTTP error responses. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Function arguments use `response.function_call_arguments.delta` and `.done`; a custom tool's input
uses `response.custom_tool_call_input.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only a suffix that may start a tool-call marker, or the tool region after one,
is held. On the unconstrained route every accepted marker form counts (`<tool_call>`,
`<function_calls>`, `<function=`/`<function `, `<invoke=`/`<invoke `, and their quoted variants),
so answer prose that writes such a tag outside Markdown code stops streaming there and the rest of
the turn arrives at the end, as ordinary text if it does not parse as calls. Constrained output
holds only at `<tool_call>`.
On the unconstrained route, malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function and custom
tools and tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management,
`access_programs`, hosted moderation, non-empty `include`, background execution, compaction,
files/audio, and OpenAI-hosted/MCP tools. These are compatibility boundaries, not silently
accepted placeholders.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE. Consecutive User or Assistant messages are joined without adding separators.
System messages in `messages` may appear anywhere, including first, between two User messages,
directly after an Assistant message, or last; each renders as its own System turn at that position
after the top-level `system` text, so appending one keeps the earlier rendered prompt reusable. The
one excluded position is between an Assistant `tool_use` and the User message carrying its
`tool_result`, which is rejected as invalid tool history. A final text-only Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media, Thinking, or tool calls and cannot start with Thinking
enabled.

`max_tokens` is optional for local clients and otherwise uses the
[default output limit](#default-output-limit); a positive
value is the complete output budget. `max_tokens:0` is rejected because NInfer does not expose a
completed zero-output cache-prewarm lifecycle. `temperature`, `top_p`, `top_k`, and
`stop_sequences` enter Engine execution, as does the NInfer `relaxed_drafting` extension
([Relaxed drafting](#relaxed-drafting), which changes the output distribution). A matched custom stop is returned as
`stop_reason:"stop_sequence"` together with the actual `stop_sequence`; context exhaustion returns
`model_context_window_exceeded`.

Thinking supports `disabled`, `adaptive`, and `enabled`; `between_tools` returns
`thinking_type_not_supported`. Enabled Thinking requires
`budget_tokens >= 1024` and less than `max_tokens`, and that budget is passed to Engine. Visible
Thinking is returned with an opaque local signature; SSE emits its `signature_delta` before
closing the block. Assistant Thinking blocks must be passed back unmodified with that signature;
signatures belong to the current serve process and are invalid after it restarts.
`display:"omitted"` is rejected because NInfer cannot provide Anthropic's
encrypted hidden-reasoning restore semantics. `preserve_thinking` remains a NInfer extension for
closed-turn reasoning history, and `graft` selects a [prompt graft](#prompt-grafts). `output_config.effort` passes its protocol-validated value to the
selected template. `output_config.format` accepts JSON Schema output as described in
[Output constraints](#output-constraints).

User-defined tools support `name`, `description`, object `input_schema`, `input_examples`, and
`strict`. `tool_choice` accepts `auto`, `none`, `any`, or named `tool`; `disable_parallel_tool_use`
enforces a single-call limit. See the common tool contract above for schema and framing details.
Deferred tools, tools that exclude direct model calls, Anthropic-provided/server tools, toolsets,
MCP, and containers remain unsupported. `tool_result` preserves text/image order and marks
`is_error:true` explicitly in the model prompt. For a visible Assistant tool-use turn, the next
User turn must provide exactly one leading result for every declared ID; valid results are matched
by ID and normalized to call order. A history that begins with results remains valid as a truncated
or imported conversation.

Ephemeral `cache_control` on the request, tools, System blocks, and User text/image frontiers is a
best-effort retention hint. NInfer maps representable breakpoints to exact prompt frontiers, keeps
the latest markers allowed by the Engine configuration, and ignores TTL and unrepresentable cache
hints rather than rejecting generation. Reuse still requires exact rendered-token compatibility;
aggregate usage reports verified reused tokens in `cache_read_input_tokens` and leaves cache
creation unknown. Streaming emits `message_start` after Engine admission commits the prefix
selection and before transfer/prefill output, so its uncached/cache-read split is already exact;
terminal cumulative usage matches the aggregate response.

Documents, Search Results, Files, server-tool results, container uploads, and
other execution-dependent blocks are rejected with the missing capability identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages; output-only sampling and
streaming fields do not affect the count:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Prompt grafts

A prompt graft is a hidden conversation prefix, for example a system turn plus an assistant
acknowledgement, which the operator loads at startup and a request selects by name. Grafts are the
[phantom-kv](https://github.com/lordx64/phantom-kv) `prefill_kv` container (format_version 1): a
safetensors file plus a `.json` sidecar beside it.

```bash
ninfer-serve model.ninfer --graft v1=C:/grafts/v1_q38_nf4.bin --graft red=C:/grafts/red.bin
```

A request selects one with the top-level `"graft": "NAME"` field on OpenAI Chat Completions,
Responses (create and input token count) and Anthropic Messages. A name the server did not load
fails with `400 unknown_graft`, and a non-string value fails as a malformed `graft` field.

`--default-graft NAME` makes one loaded graft the default for requests that state none. It must
name a `--graft`, or the server refuses to start.

| request `graft` | without `--default-graft` | with `--default-graft D` |
|---|---|---|
| absent or `null` | none | `D` |
| `""` | none | none (explicit opt-out) |
| `"X"` | `X` | `X` |

A grafted request runs exactly as if the graft's hidden turns preceded its own messages. The
graft's tokens occupy positions `[0, n)` and the request's own rendered prompt starts at `n`. With
greedy decoding, output is identical to sending those turns as literal messages. The rendered
prompt should carry no system turn of its own, because the graft already holds one. Graft tokens
count toward `--max-context` and the reported prompt/input tokens.

The container also stores the cache state that phantom-kv computed for the graft (attention K/V
and Gated DeltaNet conv/recurrent state). NInfer does not inject that state. It replays the
graft's own token ids through its own prefill, which is exact for the loaded weights and KV
storage and covers every layer, including the MTP draft layer. The end of the graft is offered to
the shared-prefix cache as a save opportunity, so conversations after the first can reuse the
whole graft instead of re-prefilling it; whether it is kept follows the
[retention rules](maintainer/resource-scheduling-and-context-cache.md#retention).

Startup validates each graft against the loaded model and refuses to start on any mismatch:
- the per-layer attention/linear-attention layout;
- the KV-head, conv and recurrent-state geometry;
- the sidecar's payload sha256;
- the replay ids against the vocabulary.

Trained `softprompt_kv` and `direct_kv` grafts carry no replayable token ids. Their stored K/V and
Gated DeltaNet state are instead installed at startup as a pinned context-cache checkpoint, and
grafted requests start from it. The K/V are written through the same append Op prefill uses, so
every `--kv-dtype` stores what prefill would store for the same rows. Because nothing is replayed,
these grafts cover the text layers only: the graft carries no draft-backend state, so under `--spec`
the MTP or DFlash context over the graft's positions is zero-filled and the draft proposes without
graft context there. Output is that of the target over the installed state, because the target
verifies every proposal; only the acceptance rate can fall. Each one holds a Device StateImage and
its KV pages for the life of the server; startup adds them on top of `--device-state-slots` and the
planned KV capacity, and a disabled context cache (`--no-prefix-reuse`) refuses them. Grafted
requests use the context cache like any other: the pinned checkpoint is the root, it is never
evicted, and later turns and shared prefixes are captured after it. In the prompt the graft's
positions are held by ids derived from the container's sha256, so the checkpoint is only ever
matched by requests using the same graft.

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health`, the [dashboard](#dashboard) page (`GET /dashboard`, which holds
no data and sends the key on its own requests) and CORS preflight requests remain unauthenticated;
`GET /v1/load`, `GET /slots` and `GET /metrics` require the key.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

The key is compared in constant time, so response times say nothing about how much of a guess was
right.

### Host and Origin checks

These checks run before every route, `GET /health` and the loading `503` included. They stop a web
page in a browser on this machine (or on the LAN) from using the server behind your back.

- **Host**, always. A request whose `Host` header names something the server does not answer to is
  refused. This is what defeats DNS rebinding, where an attacker's domain is made to resolve to
  `127.0.0.1`. Accepted:
  - `localhost`, `127.0.0.1` and `[::1]`;
  - the `--host` address or name, unless it is a wildcard (`0.0.0.0`, `::`);
  - the machine's host name, when it is a wildcard;
  - each `--allowed-host NAME` (repeatable);
  - any IPv4 or bracketed IPv6 address, such as `192.168.1.5:8080` from another machine on the
    LAN, because a rebinding attack needs a name.

  Names are matched in any case and with or without a trailing dot. The port is not compared, so
  SSH tunnels and port forwards keep working. A request with no `Host` header at all (HTTP/1.0
  tools) is accepted.
- **Origin**, only when the server has no `--api-key` and no `--cors`. A browser request from
  another site carries that site's `Origin`, and is refused unless the origin is the server's own
  (`scheme://name:port`, a name from the list above on the server's port). Clients that are not
  browser pages (`curl`, the OpenAI and Anthropic SDKs, a chat UI's server-side backend) send no
  `Origin` and are not affected. `Origin: null`, sent by `file://` pages and sandboxed frames, is
  refused, as are browser-extension origins. Give such a client a key, or start the server with
  `--cors`.
- **Sec-Fetch-Site**, under the same condition. A request a browser marks `cross-site` (a link,
  an image or a `no-cors` fetch from another site, some of which carry no `Origin`) is refused,
  so a page elsewhere cannot make the router load, wake or evict a model. `same-origin`,
  `same-site`, `none` (an address typed by the user) and no header at all pass.

A refused request gets `403` in the error shape of the endpoint it called: the Anthropic shape (type
`permission_error`) under `/v1/messages`, and the OpenAI shape (code `host_not_allowed`,
`origin_not_allowed` or `cross_site_request`) everywhere else.

```json
{"error": {"message": "Host 'evil.example:8080' is not a name this server answers to; ...",
           "type": "invalid_request_error", "param": null, "code": "host_not_allowed"}}
```

The router runs these checks on its public address and drops `Origin` (with the credentials) from
the requests it relays to its models. Each model's own server listens on loopback with no key, no
`--cors` and no `--allowed-host`, whatever the command line gives the router, so a browser page
cannot reach it directly around the router's key.

**Serving on a LAN.** Bind a wildcard or the LAN address, and give it a key:

```bash
ninfer-serve model.ninfer --host 0.0.0.0 --api-key "$(openssl rand -hex 24)"
```

Clients reach it at `http://<lan-ip>:8080` or `http://<machine-name>:8080` with no extra setup. For
any other name (a DNS alias, `gpubox.local`, a Docker Compose service name), add
`--allowed-host gpubox.local`. The value is a host name or an address, with no scheme, port, path
or wildcard; anything else is refused at startup. Without `--api-key`, a server listening beyond
loopback logs a warning at startup: anyone who can reach the address can use the model.

**Behind a reverse proxy.** Caddy and nginx (with `proxy_set_header Host $host`, which most
configurations use) pass on the Host the client used, so add each public name with
`--allowed-host`. A web UI served from the proxy's own origin sends `Origin: https://name` (port
443), which can never be the server's own origin, so a proxied server that browsers reach needs
`--api-key` (the UI sends the key) or `--cors`; a keyless server without `--cors` refuses those
requests.

## Server options

The table lists executable defaults. The startup example selects a long-context FP8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--allowed-host NAME` | one more name the [Host check](#host-and-origin-checks) accepts; a host name or address without a scheme, port or wildcard; repeatable | none |
| `--model-id ID` | override the public OpenAI model alias | artifact `metadata.name`, or architecture name |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--fit-context MIN` | treat `--max-context` as a ceiling: start at the largest context (multiple of 1,024, KV pool equal to it) that fits free GPU memory less 256 MiB, never below `MIN`; requires `--kv-capacity` unset or equal to `--max-context` | off |
| `--max-concurrency N` | resident execution lanes; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--admission-fit-check` | queue a fresh request whose whole prompt cannot fit in the KV pool beside the resident requests, instead of starting it, pausing it when the pool fills and recomputing its history; always admits when nothing is resident (counted by `ninfer_admission_fit_deferrals_total`) | off |
| `--admission-affinity` | when a lane is free, let a waiting request whose prompt resumes a prefix already held on the GPU (a cached checkpoint it may take over, or one its own session left, with no copy from Host memory) start ahead of older waiters; each waiter can be passed this way at most four times, separately from the ordinary bypass allowance. Only the eight waiters behind the head are considered (counted by `ninfer_admission_overtakes_total`). Qwen3.5 and Flash-Next; no effect with `--no-prefix-reuse`. Not run on a GPU yet | off |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `600000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--progress-anchor-tokens N` | during a long prefill, keep a private checkpoint every N tokens (a multiple of `--prefill-chunk`; needs prefix reuse; each request keeps its newest four) so a paused request that loses its snapshot replays from the nearest checkpoint instead of from the start. Each checkpoint costs one State image; Qwen3.5-family models only | `0` (off) |
| `--progress-anchors-shared` | with `--progress-anchor-tokens`: publish the anchors as shared checkpoints, so a request cancelled part-way (client timeout or disconnect) leaves its work for the retry or for any request with the same prefix. A request keeps its two newest anchors and releases an older one unless something has reused it. Not run on a GPU yet | off |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8, Linux; see the [README](../README.md#several-gpus-pipeline-stages---devices-ab)); overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|nvfp4\|k8v4` | KV-cache storage. `rk8v4` is opt-in RotorQuant and `rk4v4` opt-in Lloyd-Max 4-bit keys; all seven are accepted on this fork's sm_86/sm_89 targets | `bf16` |
| `--ngram-volume`, `--expert-state`, `--no-expert-state`, `--default-expert-state`, `--vram-headroom-mib`, `--ram-headroom-mib`, `--expert-cache-mib`, `--shared-expert-stream`, `--mtp-constrained`, `--mtp-joint-acceptance`, `--ngram-prefetch`, `--hot-embedding-rows`, `--gdn-head-map`, `--moe-precast-input`, `--moe-gate-up-registers`, `--moe-fused-route`, `--cpu-verify-misses`, `--prefill-cpu-share`, `--cpu-verify-workers`, `--gdn-async-commit`, `--hc-int8`, `--hc-fused-read`, `--ngram-flag-wait`, `--partial-ram-residency` | Qwen3.8-Flash-Next only; see [Qwen3.8-Flash-Next](#qwen38-flash-next-preview) | see there |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2 | unset |
| `--lm-head-draft` | optimized proposal head; implied by `--spec`, accepted for compatibility | on with `--spec` |
| `--ngram-draft-tokens 0\|15` | n-gram copy drafting beside `--spec dflash2`: when the text being written already appeared in the request (the prompt, a tool result -- including `cat -n` numbered ones -- or the output so far), a round verifies up to 15 copied tokens instead of the draft model's proposal; verification licenses every token, so a wrong copy costs speed, never output. Copies run when a round holds one or two requests (at two only for copies of 12 or more tokens, and only while the other request, whose row pays the copy width too, has the balance for it: about 3% of its round time if it never copies); see [the CLI guide](cli.md#n-gram-copy-drafting) | `0` (off) |
| `--relaxed-drafting [P]` | [relaxed drafting](#relaxed-drafting), opt-in and it CHANGES THE OUTPUT DISTRIBUTION: under sampling, keep a draft the model gives probability `P` or more (`0..1`; `0.5` without a value) even where exact speculative sampling would draw another token. The default `P` of requests that do not set `relaxed_drafting`; `0` leaves requests exact unless they ask. Needs `--spec` (refused otherwise) | off (exact; requests asking for it are refused) |
| `--draft-context-rows N` | [draft-head context rows](#draft-head-context-rows): up to `N` (`0..2048`) ids the requests' prompts and outputs hold that the reduced draft head lacks are scored from the full output head and may be drafted; drafts only, so output is unchanged. Needs `--spec`; refused on Qwen3.8-Flash-Next | `0` (off) |
| `--ngram-min-match N` | tokens the end of the text must match before a copy is proposed, `4..64` | `12` |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)) | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--default-max-tokens N` | output limit when omitted by a request; see [default output limit](#default-output-limit) | the remaining context |
| `--max-output-tokens N` | upper bound on every request's output budget, stated or derived; see [default output limit](#default-output-limit) | none |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--reasoning-loop off\|stop\|conclude` | reasoning-loop guard for thinking-enabled requests: end the reply or close the thinking when it keeps repeating whole passages (details under [OpenAI Chat Completions](#openai-chat-completions)) | `off` |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay` | `overlay` keeps the Vision tower in pinned host memory and encodes each image inside a window borrowed from free KV pages, or from the evict-ranked text weight tail when those fall short, so `--vision` no longer reserves device memory and `--kv-capacity auto` resolves the no-vision capacity; requires `--vision` and CUDA virtual memory management | `resident` |
| `--vision-max-merged N` | merged-token budget of one media item, `[64, 16384]`; larger images and video frame pairs are downscaled at preprocessing instead of being rejected, and the overlay window is sized for it | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--model-suspend` | enable [model suspend](#model-suspend): the suspend, resume and residency endpoints. Needs CUDA virtual memory management | off |
| `--suspend-snapshot pageable\|pinned` | where the retained state waits in host memory while suspended | `pageable` |
| `--suspend-weights artifact\|host` | where a resume takes the weights from: the artifact again, or a host copy taken at suspend (weight-sized host memory, faster wake) | `artifact` |
| `--no-auto-resume` | a request while suspended fails (503) instead of resuming the model | requests resume it |
| `--prefill-stall-budget-ms N` | while another lane decodes, cut each prefill step so it is predicted to take at most N ms; see [execution behavior](#execution-behavior) | off |
| `--yield-long-prefill-ms N` | let the request at the head of the queue pause a prefill with at least N ms of work left, when its own prefill costs at most a quarter of that; see [execution behavior](#execution-behavior) | off |
| `--prefill-decode-share P` | after each prefill step, while another lane decodes, run decode rounds for P percent (0-90) of the step's time before the next step; see [execution behavior](#execution-behavior) | off |
| `--prefill-layer-slices N` | while another lane decodes, run each text prefill chunk as N slices of layers with that lane's decode round between slices (2-16); see [execution behavior](#execution-behavior) | off |
| `--sleep-idle-seconds N` | suspend the model after N seconds with no request; needs `--model-suspend` (with a router, sleeps idle models that were started with it) | off |
| `--models-dir DIR` | [router mode](#several-models-router): serve every `.ninfer` in DIR; give no artifact path | off |
| `--models-preset FILE` | router mode: INI presets, `[id]` sections with `model = PATH` and options without dashes; `[*]` applies to all | off |
| `--models-max N` | router mode: models loaded at once (`0` = no limit); the least recently used idle one sleeps or unloads to make room | `1` |
| `--no-models-autoload` | router mode: a request for a model that is not loaded fails instead of loading it | autoload |
| `--unload-idle-seconds N` | router mode: unload a model idle for N seconds | off |
| `--device-state-slots N` | extra Device StateImages beyond `max-concurrency` | `max-concurrency` |
| `--host-context-mib N` | shared pinned Host budget for StateImages, KV and pause snapshots, including in-flight destinations; refused together with `--auto-host-cache` | `8192 MiB + 8 native StateImages` |
| `--auto-host-cache` | size the Host context budget from the host memory still free once the model has loaded, for a machine that exists to serve; see [automatic host cache](#automatic-host-cache). Refuses an explicit `--host-context-mib` | off |
| `--context-nvme-mib N` | Qwen3.8-Flash-Next: keep checkpoints in a scratch file of up to N MiB on local disk (removed at exit), with the Host context budget as its RAM cache; `--host-context-mib 0` makes it disk only. See [Qwen3.8-Flash-Next](#qwen38-flash-next-preview) | off |
| `--context-nvme-dir DIR` | with `--context-nvme-mib`, the directory for the scratch file; put it on NVMe | beside the model |
| `--host-cache-reserve-mib N` | with `--auto-host-cache`, host memory left unpinned beneath what is available | `3072` |
| `--host-cache-max-mib N` | with `--auto-host-cache`, the most it may pin, applied after the reserve; for machines whose memory other tenants share | no cap |
| `--host-cache-percent N` | with `--auto-host-cache`, the most it may pin as a percentage (1-100) of the machine's total memory (or its container's limit), however much is free at startup; the smallest of this, the cap and the free memory less the reserve applies | no limit |
| `--context-store DIR` | keep retained sessions on disk so a restart or crash does not lose the context cache; see [Context store](#context-store). Needs a nonzero Host context budget or a disk tier (`--context-nvme-mib`) | off |
| `--context-store-max-gib N` | bound the store; the least recently used sessions are removed beyond it. Requires `--context-store` | half the volume's free space |
| `--context-store-ttl-hours N` | remove sessions unused this long; `0` keeps them until space is needed | `168` |
| `--context-store-idle-seconds N` | write a session unused this long, and changed since it was last written, in the background; `0` writes only at shutdown | `30` |
| `--context-store-restore-seconds N` | time budget for restoring sessions at start-up (only those the directory holds; a bucket's are downloaded on request), and the longest a request waits for its stored session to be read back, a download included | `120` |
| `--context-store-flush-seconds N` | time budget for writing sessions that are not yet stored at shutdown | `60` |
| `--context-store-peer DIR` | import the sessions another engine on this box keeps in its store directory, by hard link (repeatable; same volume; each engine needs its own `--context-store`); see [Two engines on one box](#two-engines-on-one-box). Requires `--context-store` | none |
| `--context-store-peer-poll-ms N` | how often peer directories are looked at (at least 1) | `250` |
| `--context-store-compress` | write new chunks zstd-compressed, in the directory and the bucket (about 8 % smaller; chunks of both forms are always read); see [Context store](#context-store). Requires `--context-store` | off |
| `--context-store-s3-endpoint URL`, `--context-store-s3-bucket NAME` | keep a copy of the store in an S3-compatible bucket (both required together; credentials from `NINFER_S3_ACCESS_KEY_ID` / `NINFER_S3_SECRET_ACCESS_KEY` or the `AWS_` equivalents); see [Keeping a copy in a bucket](#keeping-a-copy-in-a-bucket). Requires `--context-store` | off |
| `--context-store-s3-prefix P`, `--context-store-s3-region R` | key prefix inside the bucket, and the signing region | none, `us-east-1` |
| `--context-store-s3-upload-delay-seconds N` | how long a written session waits before it is uploaded; a newer write of the conversation within it replaces it, so a burst of turns is uploaded once. Shutdown uploads at once | `30` |
| `--no-exit-on-engine-failure` | stay alive (answering 503) when the engine latches unavailable, instead of logging FATAL and exiting with status 3 after 5 s | exit |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--tolerant-tool-calls` | repair broken tool-call markup on `tool_constraints:"auto"` requests instead of returning it as text; never keeps a call whose argument was cut. See [Tolerant tool calls](#tolerant-tool-calls) | off |
| `--graft NAME=PATH` | load a [prompt graft](#prompt-grafts) a request may select by name; repeatable | none |
| `--default-graft NAME` | apply a loaded graft to requests that state none; `"graft": ""` opts out | none |
| `--reasoning-effort minimal\|low\|medium\|high\|xhigh\|max` | effort for thinking-enabled requests that state none | template default |
| `--cors` | permissive browser CORS headers | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..20`; zero selects the top-20 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup; the operational context-cost record and JSONL `server_start` identify the
selected source.

Engine selects sampling defaults from the loaded architecture and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`.
Host state and Main/Backend KV share one startup-fixed byte budget; this is context storage, not a
limit on total process RAM. `--host-context-mib 0` disables Host context backing.
`--no-prefix-reuse` disables cross-request history reads and writes; pause/replay recovery remains
available, and the capacity flags may still be specified.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Every line is one `ninfer_serve_request_log` schema-v30 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | build version, artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, unified Host context capacity and occupancy, KV sizing ledger, CUDA Graph allowance, CUDA/GPU environment, redacted `argv`, `effective_argv` (a `--config` file's words, then the command line's) and `setting_sources` (each option given: `cli`, the config path, or `default`) |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, preemption/recovery counters, thinking-budget application counters, model thinking tokens, reasoning-loop guard firing, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters |
| `request_scheduling` | request identity, pause/restore/recovery transitions, Snapshot revocation, Engine observation time and cumulative global/request work counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, repeated parameter names resolved to their last value
(`duplicate_parameters_repaired`), and a stable text-fallback reason. Fallback reasons are `none`,
`malformed_structure`, `invalid_tool_name`, `undeclared_tool`, `trailing_content`, and
`quoted_in_code` (a constrained `auto` call opened inside Markdown code; see
[Tool constraints](#tool-constraints)).
`recovered_from_reasoning` is true when the structured calls came from a turn that ended inside
its unclosed thinking. `tolerant_recovered` is true when `--tolerant-tool-calls` repaired markup
the strict parser returns as text to produce the structured calls (see
[Tolerant tool calls](#tolerant-tool-calls)); the fallback reason is then `none`. These counters
contain no tool arguments or generated text. `server_start.server.tolerant_tool_calls` records the
flag.

`request_done.constraint` carries the same constraint observation as the HTTP terminal result,
or `null` for unconstrained requests. Preparation failures and execution errors use the existing
rejection/error records rather than successful constraint outcomes.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, `accepted_per_position` (draft-model rounds
only), `ngram_draft_tokens` (the copy window, 0 when off), and `ngram_rounds`, `ngram_drafted_tokens`
and `ngram_accepted_tokens`, the copy rounds, which are a subset of the round and token counts above.
The stderr request line adds `copies accepted A/D (P%)` when the request verified copies. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.
`generation.scheduling` records preemptions, snapshot/replay restores, replayed tokens, paused time
and request-owned transfer bytes. Replay rebuilds committed state without adding new output usage.

`generation.admission` records the initial `preferred_reused_tokens`, `source_wait_seconds`,
`revoked_checkpoints`, and `fallback_reason`. Source waiting is a subset of initial queue time;
selecting or retaining a checkpoint does not itself count as a cache hit. Revocations count retained
checkpoint references removed under resource pressure. Fallback reasons are `none`, `source_invalid`,
`source_revoked`, `cost_changed`, `capacity_limit`, and `isolated_capacity`.

`request_scheduling` records `pause_started`, `paused`, `restore_started`, `restored`,
`replay_complete`, `recovery_complete`, `snapshot_revoked`, and a `terminal` boundary for preempted
requests. `preemption_index` identifies each pause cycle; `route` is `snapshot`, `replay`, or `null`
while pause preparation has not yet selected the saved representation. `steady_ns` is captured on
the Engine worker, and `elapsed_ns` starts at Engine submission; JSONL writes happen on the request
consumer thread. Compare `steady_ns` rather than delivery order across requests. `restored` ends
binding; `recovery_complete` marks the first fresh committed unit or normal terminal progress,
not merely rebuilding the old frontier. Cancellation can end the cycle without that event.

Each event's `progress` carries global and request-owned prefill, decode/control and replay token
counters. Between two boundaries, subtract the request delta from the global delta to measure
other requests' completed work. In particular, `restored` to `replay_complete` establishes whether
other work advanced during Replay without relying on periodic scheduler gauges. Events are enabled
only with request logging and add no per-token records.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the aggregate prompt and maximum-item token
bounds plus encode peak and handoff layout/usage within that same allocation; these bytes must not
be added to `workspace.capacity_bytes`. The field is `null` when Vision is disabled.

`request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
`device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
`host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
`engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait, and
round count; `units` reports its prefill/control unit counts. In a compact batch every participating
request is delayed by the full round, so these values explain request latency but **must not be
summed across concurrent requests**.
`constraint_draft_wait_exposed_seconds` is the request's exposure to the batch's draft-ready wait,
already included in `device_wait_exposed_seconds`. The `throughput.host_work.constraint_draft_wait_seconds`
interval and Prometheus counter count each batch once.

`request_done.first_output_timing` freezes observations immediately before Engine publishes its
first nonempty output delta. It is `null` when no such output exists. This boundary differs from the
first accepted model token and the client's first HTTP output. Engine elapsed time begins at submit:
initial queue ends when the successful binding attempt starts, initial binding ends when the
binding is installed, and paused time includes pause preparation, waiting and restoration. The remaining
interval is resident time. Its `engine` observations describe resident Host/Device-wait exposure;
`prefill` and `replay` describe this request's submitted work. Terminal `engine_timing` still covers
the whole request.

The work `gpu_seconds` measures Text prefill stream intervals, including their MTP/DFlash work;
Vision encode, standalone bridges and exact-hit sampling fall outside that interval.
`context_transfers` reports completed request-owned State/Main KV/backend KV copies by direction,
using transfer-event time and bytes. GPU intervals overlap Host submission and waits, so they are
separate evidence, not additional wall-time stages. Background reclamation remains Engine-wide.

`result.generated_token_ids` records committed output token IDs for exact prefix analysis.
The JSONL file never records an API-key value; `argv` and `effective_argv` replace it with
`<redacted>`.
Operational stderr summaries are rounded and are not the
aggregation source. OpenAI Responses, OpenAI Chat, and Anthropic generation requests receive a
request ID when they enter synchronous preparation. Successful preparation produces
`request_start`; a preparation failure produces `request_rejected` without a matching start. Each
started generation transaction then has exactly one machine terminal: `request_done` when Engine
returns its outcome, or `request_error` when generation fails before an outcome exists. Later
response rendering, Responses storage, or terminal transport failures are operational response
events only and do not add a second JSONL terminal. Schema/model validation rejections before
preparation and token-count-only calls are not measurement requests and do not receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `paused`, `replaying`, `materializing`,
`capture_pending`, and `terminal_pending` fields are the Engine scheduler snapshot at the end of the
interval. The JSONL `context_cache` object reports selections, captures, StateImage operations,
transfers, tail-page COW and pressure spills as interval deltas; `occupancy` and `last_selection` are
end-of-interval gauges. The separate `scheduling` object reports preemptions, restores and replayed
tokens. Occupancy includes Host reservations while transfers are in flight.
The top-level `engine_recoveries` field is the interval delta of the worker-recovery counter. It is
not part of `context_cache`, because a recovery is an engine-wide event (see
[Startup readiness](#startup-readiness)).

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose stats-publication work; these detail values
are already contained in a top-level Host phase and must not be added to `total`.
Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. Pretty throughput contains nonzero token rates and counts,
the current running/prefill/decode-ready composition, nonzero waiting/materialization/terminal
states, average decode batch, and Host-active time plus its fraction of the interval. Use JSONL for
complete measurement analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.

## Execution behavior

The server owns one resident Engine with `1..8` execution lanes fixed at startup. At each decode
boundary, eligible decode-ready requests form one compact batch, processed by one model traversal
and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
Once admitted, a request may pause for resource pressure without restarting this initial-admission
deadline. Fresh requests enter in FIFO order with a bounded bypass allowance when an earlier request
cannot fit. With `--admission-affinity`, a request that resumes a prefix still held on the GPU may
also start ahead of an older request that could have started, at most four times per older request.
There is no admission ETA or unbounded overflow queue. A fresh request never preempts a resident
one, so when every lane is busy a queued request waits out the generations ahead of it, and the
deadline has to be scaled to the longest response the deployment allows rather than to a connection
timeout: at C1 on an RTX 3090 a single 6,500-token response occupies the engine for about 106
seconds. The 600,000 ms default admits a queued caller behind roughly ten such responses; lower it
only to fail fast on purpose.

Each worker cycle runs the permitted control and decode work and then one prefill or replay chunk.
Prefill chunks rotate among resident requests, so a short prompt that already holds a lane is not
held behind another lane's long prefill. A request that has no lane yet still waits for a resident
to finish, be cancelled or pause, and a smaller chunk does not admit it sooner. `--prefill-chunk`
therefore sets the worst-case pause every active stream sees while a prompt is ingested. A decode
round takes tens of milliseconds and a prefill chunk hundreds, so a stream gets about one token per
chunk while another request prefills. Measured on an RTX 3090 with the 27B on the previous engine,
which alternated one chunk with one decode round in the same way (not re-measured on this engine):
a 24.7K-token prompt prefilling at chunk 1024 took a concurrent stream from about 48 tok/s to
1.4 tok/s (median gap 725 ms), and with `--prefill-cublas --prefill-chunk 4096` to 0.75 tok/s
(1.6 s gaps). Ingesting a 4,900-token prompt behind four active streams, the largest inter-token gap
measured 1,043 ms at chunk 1024, 515 ms at 512, and 312 ms at 256, against an 82 ms median decode
interval; the ingesting request's own prefill rate fell only from 1,135 to 1,130 to 1,110 tok/s.
Prefill units of different requests are never batched together, so a smaller chunk trades almost no
ingestion throughput for a proportionally smaller stall. The shipped concurrent launcher uses 512.

`--prefill-stall-budget-ms N` makes that choice per step instead of per server. While another lane is
decoding, the Engine cuts a prefill step so the context-cost model predicts it to take at most N ms,
in multiples of 128 tokens and never below 512; with nobody decoding a step keeps the full
`--prefill-chunk`, so a lone long prompt prefills at full speed. The prediction is the machine's
fitted per-step, per-token and per-attention-pair costs, scaled by how long recent steps actually
took against what the model said, so it follows a card that runs slower than the preset. A step deep
into a long prompt costs more attention than an early one and is cut harder for the same budget.
A smaller step is slower per token: at `--prefill-cublas` the weight dequantise pass is paid once per
step: measured end to end on the 27B, the route runs 1.22x the integer route's speed at chunk 512 and
1.59x at 4096 (`src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.h`). Pick a chunk near the largest you
want and let the budget shorten it only when somebody is waiting. `ninfer_paced_prefill_steps_total`
and `ninfer_paced_prefill_deferred_tokens_total` count the steps it cut and the tokens it pushed to
later steps. Off by default; it does nothing at `--max-concurrency 1`, where no other lane decodes.
Qwen3.8-Flash-Next sizes its own prefill calls and refuses the flag.

`--prefill-layer-slices N` (2-16) is the option that shortens the gap without shrinking the chunk. While
another lane is decoding, each text prefill chunk runs as N slices of consecutive layers, and the
Engine runs that lane's decode round between slices: its next token waits one slice, about a step's time
divided by N, instead of the whole step, and the chunk keeps its size, so the cuBLAS route's weight
dequantise pass is still paid once per chunk. That is what separates it from a smaller `--prefill-chunk`
or a stall budget, which buy the same gap with the prompt's own speed. What it costs: two copies of the
residual stream per slice boundary (a few dozen MB at 4,096 tokens on the 27B, well under a millisecond),
one extra decode round per boundary while the neighbour lasts, and a delay for anything that needs the
prefill path while a chunk is part-way: a thinking-budget control batch waits until the chunk is done, a
lane that wants to pause the prefill (yielding, or resource pressure) waits for the chunk boundary, and
another prompt's prefill does not start until the chunk in progress has finished. Cancelling a request
part-way is immediate. A chunk with media, with a DFlash backend, or on a pipeline split runs whole, so
the default one-lane DFlash2 profile is unaffected; the two-lane MTP profile is the one it is for.
`ninfer_sliced_prefill_steps_total` and `ninfer_prefill_slice_boundaries_total` count the steps run
sliced and the boundaries where the neighbour decoded. It combines with the stall budget (the budget
cuts the chunk, the slices divide what is left) and the decode share. Off by default; Qwen3.8-Flash-Next
refuses the flag. Run on an RTX 3090 (2026-10-22, Qwen3.8-27B, a stream beside a 108K-token uncached
prompt at chunk 4096): the neighbour's worst gap fell from 2.93 s to 402 ms at `--prefill-layer-slices 8`
while the prompt took 61 s instead of 56 s (a 500 ms stall budget gets 528 ms at 72 s). Greedy output of
a sliced prefill matched the unsliced run token for token in the engine test
(`sliced_prefill_matches_whole`) and in a served 56K-token prompt with a stream and a short request
beside it.

`--yield-long-prefill-ms N` is for the queue rather than the other streams. A fresh request never
preempts a resident one, so with every lane busy it waits for the lane's request to finish, which behind
a 100,000-token prompt is about a minute however small the newcomer is. With the flag, the request at the
head of the queue may pause a resident prefill that still has at least N ms of work left, as the cost
model predicts it, when the newcomer's own prefill (uncached, so a cache hit only makes it cheaper) costs
at most a quarter of that. The prefill is paused at a step boundary and keeps its progress as a snapshot;
it is restored behind the newcomer, once that request has been admitted (or after 10 s if it cannot be).
Several limits keep it from doing harm: a pause that cannot save the prefill's progress is not made (the
alternative would recompute it all), a prompt with media is never paused, a request is paused at most
four times, and the same newcomer never pauses a second prefill. `ninfer_prefill_yields_total` counts the
pauses. It needs a free state slot and Host room for the snapshot (the same capacity as a pause for
resource pressure), and it does nothing when a lane is free. Off by default; Qwen3.8-Flash-Next refuses
the flag.

`--prefill-decode-share P` (0-90) is the other half. After each prefill step, while a lane is decoding,
the Engine runs decode rounds until they have used P percent of the time the step took before it runs
the next step. At 50 a 1.5 s step is followed by 1.5 s of decode rounds, so each stream gets a burst of
tokens (DFlash2 and MTP rounds emit several each) instead of one, and the prompt takes about twice as
long while the neighbour lasts. It does not shorten the longest gap between a stream's tokens, which is
still one step; use the stall budget for that. The two combine: a budget of 500 ms and a share of 50 give
a stream a burst every ~1 s with half its normal speed. The debt a step creates is capped at 10 s, and
after 4,096 deferred turns in a row a step runs regardless. `ninfer_deferred_prefill_turns_total` counts
the cycles that held the prefill back. Off by default.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media preparation uses a shared permit pool sized from `--media-live-mib`
and the maximum supported prepared-payload size per request. Waiting media requests retain the same
cancellation and timeout deadline. Model output is bounded by the same finite request count and
each request's effective output-token limit; output callbacks and network serialization run
outside the GPU executor and do not delay formation of the next batch.

`--max-context` is each sequence's logical ceiling. `--kv-capacity` fixes the shared Main Text KV
pool used by active requests and retained prefixes. `auto` accounts for the complete enabled runtime
and leaves 1 GiB of sizing headroom; omitting the option makes it follow `--max-context`. Capacity
resolves once at startup.

`--fit-context MIN` makes `--max-context` a ceiling instead of a demand: after the weights load, if
the plan at `--max-context` does not fit the free device memory less 256 MiB, the server plans
smaller contexts with the same planner and starts at the largest multiple of 1,024 tokens that fits,
with the KV pool equal to it, logging `context lowered to N tokens (asked for M)`. `/v1/models`,
`/props`, request limits and `/metrics` report the context it started with. Below `MIN` it refuses
to start (`not enough free GPU memory to start: ...`). It needs `--kv-capacity` unset or equal to
`--max-context`; the launchers' default profiles pass `--fit-context 32768`.

Before each prefill, decode or replay unit, the runtime reserves the additional pages and temporary
storage required by that unit. It first reclaims inactive cache resources when capacity is short.
If resident requests still cannot advance together, it pauses a younger request while preserving
progress for the oldest resident request. A paused request does not block fresh requests that fit
the remaining capacity. Restoration follows original request order and reserves enough space to
rebuild the saved frontier and complete one new execution unit.

A paused request keeps its committed output and protocol state. With sufficient Host backing, it
can restore a snapshot; otherwise it rebuilds model state from retained input and committed tokens.
Replay does not resample or republish those tokens, but it consumes compute and can increase gaps in
the output stream. The policy and ownership rules are defined in
[Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The pretty completion record
shows `cache N (P%, path)` using readable path labels; JSONL retains the exact
`prefix_cache_hit_tokens` and `prefix_reuse_path` fields (`root` or `checkpoint`). Reuse requires
matching KV, recurrent state, hidden state, selected-backend state and exact prefix identity.

Completed conversation endpoints serve direct continuations. A separate input checkpoint preserves
the stable boundary before a response that a later prompt may normalize or replace. With
`preserve_thinking=true`, that boundary precedes the response's generation prologue; with `false`,
it precedes the assistant turn whose closed reasoning may be omitted. The next request can therefore
recompute the changed suffix while retaining the preceding conversation. Capture follows these
semantic boundaries and shared-prefix hints.

Changing `preserve_thinking` or reasoning effort changes the rendered prompt where applicable;
already retained exact prefixes remain usable. An appended mid-conversation system message is an
ordinary suffix. Modifying, removing or moving a historical message changes the prefix and may
require an earlier checkpoint or a root prefill.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools or enforce tool-argument schemas through constrained decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.

### Relaxed drafting

`--relaxed-drafting [P]` (off by default) and the request key `relaxed_drafting` give up exact
speculative sampling for speed. **They change the output distribution.** With them off, speculative
decoding (MTP, DFlash, DFlash2, n-gram copies, and Qwen3.8-Flash-Next MTP) keeps a draft only as
the exact rejection rule allows, so the output has the model's own sampling distribution. With a
threshold `P`, a sampled request also keeps a draft whenever the target gives it probability `P` or
more under the request's final distribution (after the penalties, any token mask, `top_k`, `min_p`,
`top_p` and `temperature`), even where exact sampling would have drawn another token. At such a
row the output is the draft every time instead of a draw: that row is greedy, its total-variation
distance from the model's distribution is `1 - p(draft)`, and every other row is unchanged. The
logits are unchanged, so a teacher-forced perplexity or KLD check cannot see this change.

What stays exact:

- greedy requests (`temperature: 0` or `--greedy`): nothing is drawn;
- constrained requests (`response_format` JSON or schema, `structured_outputs`, a required or named
  `tool_choice`): their grammar rows are never relaxed;
- every token inside a tool call (from `<tool_call>` to `</tool_call>`), so a tool's name and
  arguments are sampled exactly: a request whose committed text ends inside an open call is not
  relaxed, and no position at or after a drafted control token, think or tool tag (the tokenizer's
  added tokens) in a round is relaxed;
- `logit_bias`: nonzero values are refused on Chat Completions, so no biased request reaches it.

The server option sets the default threshold and turns the relaxed acceptance kernels on; with it
off (the default) only the exact kernels run and a request with `relaxed_drafting` above 0 fails
with 400 `relaxed_drafting_not_enabled`.
A request overrides the default with a number in `[0,1]`, `true` (0.5) or `false` (exact); other
values fail with 400 `invalid_relaxed_drafting`. `--relaxed-drafting 0` keeps requests exact unless
they ask. The startup refuses the option without `--spec`, and with a tokenizer whose added tokens
are not one contiguous id range or whose tool tags are not single added tokens.
`ninfer_spec_decode_relaxed_kept_tokens_total` counts the drafts it kept.

Basalt, which introduced the rule, measured on its own engine and model (an RTX 5090; the numbers
are context, not NInfer measurements, and nothing here has been measured on an RTX 3090): at
`P = 0.5` about +3 to +7% tokens per window for a total-variation cost of 0.02 to 0.04 per token,
at `P = 0.3` +6 to +12% for 0.04 to 0.07, at `P = 0.2` +7 to +15% for 0.05 to 0.09, roughly +1.5%
per 0.01 of total variation at every threshold; decode speed +3 to +20% at temperature 0.6 and 1.0;
no measurable loss on gsm8k and MMLU-Pro at 300 and 280 items, and tool calls with declared names
only. The cost is diversity, which those tasks do not measure.

### Draft-head context rows

The Qwen3.8-27B and Qwen3.6-35B-A3B drafters (MTP, DFlash and DFlash2 with the optimized proposal
head that `--spec` loads) score only the head's 131,072 most frequent tokens, so a name or
identifier outside them can never be drafted, however often the request repeats it.
`--draft-context-rows N` (off by default; `N` up to 2048) keeps up to `N` token ids that the running
requests' prompts and committed outputs hold and the proposal head lacks, one set shared by every
lane, and scores them from the full output head beside the proposal head: MTP and DFlash take the
best of both, DFlash2's sixteen candidates may include them. Ids are noted as they enter a request
(its prompt at admission, its output as it is committed) and leave with the request; once `N` slots
are taken, new ids wait for a free slot. **Drafts only**: verification decides every token, so the
output is unchanged (greedy and exact sampling alike); only acceptance, and the time the extra rows
take, change.

The rows are read from a Q8, Q6 or Q4 row-split output head (the stored head, `--lm-head-q6` or
`--lm-head-q4`); startup refuses the option with another head format, without `--spec`, and on
Qwen3.8-Flash-Next, which drafts with its full output head already. No RTX 3090 measurement has been
made yet: the gain to expect is bounded by the acceptance a full-vocabulary draft head would add
(see [performance](performance.md)).

### Automatic host cache

The Host context budget (pinned memory for StateImages, KV pages and pause snapshots once they leave
the GPU) defaults to 8 GiB plus eight model StateImages, or the explicit `--host-context-mib`.
`--auto-host-cache` replaces that with one size taken from the machine, once, after the model has
loaded:

- The available memory is the smaller of the system's available memory (`MemAvailable` on Linux,
  available physical memory on Windows) and what remains under the container's cgroup limit, so a
  rented container is sized by its own limit rather than the host's RAM. Startup fails, naming the
  explicit option, if the platform reports neither.
- The machine is assumed to serve only this process, so everything but a fixed reserve is pinned:
  `--host-cache-reserve-mib` (default 3072) is left for what still grows after sizing: CUDA and
  cuBLAS host state, the tokenizer and server, request buffers and the Responses store (256 MiB by
  default). On the 27B that growth measured about 2 GiB after sizing, and two short requests added
  about 30 MB; the default adds margin for long prompts. Pinned pages cannot be reclaimed, and in a
  container exceeding the limit kills the process, so lower the reserve only after watching the
  process's memory under your own load.
- `--host-cache-max-mib` caps that budget. Without it, a machine with a lot of free memory pins
  nearly all of it, which is only safe if nothing else will want the memory later. On a rented GPU
  box the host's RAM is shared with other tenants: a 64 GB host with 29 GB in use by others left
  `MemAvailable` at about 52 GB, 49 GiB was pinned, and the engine was killed by the kernel's OOM
  killer when the neighbours grew (the container's own limit was never reached, so the reserve and
  the cgroup term could not help). Set the cap to what the workload actually uses of the Host
  context; `occupancy.host_context_bytes` in `GET /v1/load` shows how much of it is occupied.
- With vision enabled the reserve also covers the media caches, which grow after sizing and are
  bounded only by their options: `--media-cache-mib` and `--media-live-mib` (1 GiB and 2 GiB by
  default) are added to `--host-cache-reserve-mib`. Without that, a vision workload could outgrow the
  3 GiB margin, which was measured on text requests only.
- `--host-cache-percent N` is the "take half the RAM" setting: `--host-cache-percent 50` pins at most
  half of the machine's total memory (physical memory, or the container's cgroup limit when that is
  lower). It does not depend on what is free at startup, so two servers started together or a
  neighbour that was idle at launch do not change it. It only lowers the budget; the free memory less
  the reserve, and `--host-cache-max-mib`, still apply, and the smallest of the three wins.

The result is the same single shared budget an explicit `--host-context-mib` sets: StateImages, KV
and pause snapshots together never exceed it, it is pinned once at startup, and the cache reclaims
retained entries to stay inside it rather than growing. The resolved value is what the
`context cache |` startup line and the request log report. The option combines with
`--device-state-slots` but not with an explicit `--host-context-mib`.

The budget, automatic or explicit, is pinned in full on Windows as on Linux. Earlier builds clamped
it on Windows to half of the device memory left after the model and KV pool, less 1 GiB, because
WDDM once charged pinned host memory against the card; on current drivers pinning no longer tracks
free VRAM (measured on an RTX 3090, driver 616.64: 16 GiB pinned with 804 MiB of the card free), so
the clamp was removed and an 8 GiB budget really pins 8 GiB of RAM. The logged figures are the ones
pinned. The automatic sizing constants are conservative defaults chosen by reasoning, not measured
against hit rates on a live workload.

### Default output limit

A request that omits its output limit (`max_completion_tokens`/`max_tokens` on Chat Completions,
`max_output_tokens` on Responses, `max_tokens` on Messages) may generate up to the remaining context
(`--max-context` minus the prompt). `--default-max-tokens N` replaces that with a fixed cap, still
bounded by the remaining context. An explicit request limit always wins, and `--max-output-tokens N`
bounds every request, including ones that state a larger limit; the bound is not an error, the run
just ends with the length finish reasons below. The JSONL request record reports the budget actually
submitted as `requested_output_tokens`.

A run that exhausts the budget finishes with `finish_reason:"length"`, Responses `incomplete` with
reason `max_output_tokens`, or Anthropic `stop_reason:"max_tokens"`, or
`stop_reason:"model_context_window_exceeded"` when the budget was the remaining context.

An output limit does not reserve KV. A request acquires pages as it generates, so a client that
states a very large limit does not evict other sessions' retained contexts up front; when concurrent
growth exhausts the pool, the scheduler reclaims retained checkpoints first and then pauses a younger
request, as described above.
