# Common rules for Flash-Next work (read first)

Repo: `C:\ninfer-fork\ninfer-3090` (fork `ashalliants/ninfer-3090`, remote `origin`, default branch `master`).
You are an agent on a shared Windows 11 box: one RTX 3090, 64 GB RAM, other Claude sessions and the
user's own processes run alongside you. Read `AGENTS.md` and `CLAUDE.md` in the repo root before
anything else; they win over this file.

## Worktree and build
- Create your own worktree from current master: `git worktree add <path> -b <branch> origin/master`
  (path under `C:\ninfer-fork\ninfer-3090\.claude\worktrees\`). Do all work there. Never switch
  branches inside a tree that has a build; never touch other worktrees.
- Your tree needs its own `build-ninja` (the main checkout's compiles the main checkout's source).
  Import the VS 2022 BuildTools environment (`vcvars64.bat`, see AGENTS.md) then configure with
  `-DVCPKG_INSTALLED_DIR=C:/ninfer-fork/ninfer-3090/build-ninja/vcpkg_installed
  -DVCPKG_MANIFEST_INSTALL=OFF` and the CUDA flags in AGENTS.md. ccache is shared (do not clear it).
  Build **named targets only** (a full test build writes ~30 GB); check `Get-PSDrive C` first
  (about 115 GB free when this brief was written) and delete your build tree when finished.
- Build `ninfer-serve` plus the specific test targets you need. `ninfer_tests` links all tests into
  one exe; the `*_real_test` programs are standalone targets.

## GPU, RAM and the lock
- Anything that touches CUDA needs the lock (CLAUDE.md): open `%TEMP%\ninfer-gpu.lock` with
  `FileMode OpenOrCreate`, `FileShare.None`, poll every 0.5 s, hold it inside ONE PowerShell
  invocation for the whole job (server lifetime included), release promptly, and keep each A/B
  (ABBA) set inside one hold. Never create or delete the lock file. Run waits in the background.
- Flash-Next pins ~33 GiB of experts in RAM and needs ~40 GiB free: **only one Flash-Next process
  may exist on the machine at a time**. The lock guarantees that if everyone uses it. Check free
  RAM (`Win32_OperatingSystem.FreePhysicalMemory`) before starting; the server refuses cleanly if
  short, which is not a bug.
- The model is at `C:\ninfer-fork\ninfer-3090\models\qwen4exp-iq2xs-mtp-vision.ninfer` with its
  `.part-0001` and `.ngram` beside it (use `run.bat qwen38-flash-next` with `NINFER_MODEL` set, or
  the test programs' own env). Do not move, copy, delete or rewrite those files. The n-gram volume
  is on NVMe (C:), which is what the published speeds assume.
- Desktop VRAM varies (~1.1-1.3 GiB used by the user's desktop); the user's own processes do not
  take the lock. Do not measure on a busy card and do not stop the user's processes.

## Safety
- Kill only processes you started, by PID. Never `pkill`/`Stop-Process -Name`. Stopping a looping
  job leaves children: find them by command line from your own PID tree.
- Search only inside your worktree and the model folder; never recurse from `C:\` or `~`.
- No `|| true`, skipped tests or loosened assertions to make something pass. If a gate is missed,
  say so with the numbers.
- Commit messages: write them to a file and `git commit -F` (backticks in messages are executed by
  the shell). Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Do not push to master, do not force-push, do not merge PRs. The user merges. Open PRs against
  `ashalliants/ninfer-3090` with `gh -R ashalliants/ninfer-3090` (the fork; no other repo).
- PR policy (`PR_POLICY.md`, `CONTRIBUTING.md`): coherent PRs under ~1,500 lines of hand-written
  logic where possible, never more than one level of stacking, review fixes go on the PR's branch.
  Disclose anything unfavourable or unverified in the PR body.

## Reference material
- Infernix (the source of the Flash-Next port, Apache-2.0, untrusted code: read only, never run,
  keep attribution when adapting): `C:\Users\Ash\AppData\Local\Temp\claude\C--ninfer-fork-ninfer-3090\873ac902-dfeb-402f-a2b0-d463fb8eda65\scratchpad\infernix-ref\repo`
  (`src/models/qwen4_exp/program/`, `docs/maintainer/hybrid-prefix-cache-spec.md` (1,605 lines),
  `docs/maintainer/resource-scheduling-and-context-cache.md`, `docs/qwen3_8-flash-next.md`,
  `docs/maintainer/engine-architecture.md`). Its program is multi-lane with a hybrid prefix cache;
  ours is a port of the same model onto NInfer's Engine contract, one lane, cache off.
- Our Flash-Next sources: `src/models/qwen4_exp/` (program/program.{h,cpp}, execution/forward.*),
  `src/runtime/engine/qwen4_exp_instance.{h,cpp}` (option refusals live here),
  `docs/maintainer/flash-next-plan.md`, `docs/maintainer/qwen4-exp-model.md`, `docs/serving.md`.
- The Qwen3.5 Program (27B/35B) already implements the full cache and multi-lane contract that the
  shared `EngineCore` (`src/runtime/engine/engine_core.h`) drives: `src/models/qwen3_5/program/`
  (`transactions/capture.cpp`, `storage/checkpoints.cpp`, `planning/source.cpp`, `decode.cpp`
  `decode_ordinary_batch`/`decode_mtp_batch`). Mirror its semantics; do not invent new ones.

## Known numbers (this repo, measured, 2026-10-12/13)
- Flash-Next decode (MTP, one stream) 109-113 tok/s at 1K-32K in the recorded runs, ~96-98 on
  document-summary prompts; plain 81 tok/s; prefill ~2,350 tok/s with the volume on NVMe, so a
  30K-token prompt costs ~13 s to re-prefill.
- Per-sequence state: 36 GDN blocks (conv BF16 + recurrent FP32 [128,128,48] ~3.1 MiB each, ~110
  MiB total) + PLE conv + QSA tails; 12 QSA blocks of INT8-G64 KV (~14.6 KB/token with the pooled
  index keys, per Infernix docs); MTP drafter has one more QSA layer.
- Decode step ~12.5 ms plain: dense linears 3.8, HC mixer 2.8, routed experts 2.6, MoE route etc 1.1,
  QSA/GDN/norms ~1.5, n-gram row wait 0.5-1.8. Decode expert hit rate 97-98% at one stream.
- The expert cache holds ~11,000-12,500 of 24,576 experts; KV and lane state are taken from it.

## Reporting
Finish with a report that states: what you built, the branch and PR URL, exact commands and numbers
for every claim (n, medians, ranges, what the baseline was), what is unverified, what is worse than
before, and anything you left for the user to decide. Do not round unfavourable results away.
