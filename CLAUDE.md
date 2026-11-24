# CLAUDE.md

`AGENTS.md` holds the project rules and the Windows build recipe. This file adds what Claude sessions
need to share the GPU safely, and the rule below for speed changes.

## Speed changes update the published numbers

Any change that makes any model faster (decode, prefill, TTFT, speculative decoding, batching,
load time, context fit) updates the published figures for that model **in the same PR**, on every
card it was measured on (RTX 3090 or Tesla V100). A speed PR without its doc update is not done.

Update every place the old figure appears; grep for it rather than trusting this list:

- `README.md`: the model's rows, headline claims and the supported-features wording;
- `docs/performance.md`, and `docs/tesla-v100.md` for V100 figures;
- `docs/serving.md` / `docs/cli.md` when a default or a flag's effect changed;
- the model's maintainer doc under `docs/maintainer/` (its performance and status sections);
- the model card `model-cards/<repo>/README.md`. Merging to master publishes it to Hugging Face
  (`.github/workflows/publish-model-cards.yml`), so a stale card goes public;
- charts in `docs/assets/`: regenerate them, then run `tools/sync_model_cards.py` so each card's copy
  matches (`--check` runs in CI).

How to report the numbers:

- Lead with the best clean run you measured: correct output, the fastest valid settings, and no
  thermal throttling during the run (sample `nvidia-smi --query-gpu=clocks_throttle_reasons.active
  --format=csv` while it runs; discard a run that shows thermal slowdown). State its conditions (prompt type, context, flags, power
  limit), and give the typical range or session spread second.
- Every figure is a real measurement on that card at the claimed scope (serve-level for tok/s
  claims). Never extrapolate, round up, or carry a number across cards or builds.
- Remove "not supported yet" and limitation wording that the change made false.

## Build only what you need

A full CUDA build sends hundreds of kernels through `ptxas`, at 1-3 minutes each. It loads the
machine for an hour while other agents are timing runs. On 2026-11-24 a review-fix agent built the
whole `ninfer_tests` aggregate for a change to two host headers. The host-only tree builds the same
tests in minutes.

- **Host-side changes:** runtime, engine, scheduler, serve, API, tools or their tests. Build the
  host-only tree, as CI's `host-tests` job does, and run its tests. No `nvcc` is involved:
  ```
  cmake -S . -B build-host -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_HOST_ONLY=ON -DBUILD_TESTING=ON
  cmake --build build-host --parallel 8
  ctest --test-dir build-host --output-on-failure
  ```
- **Kernel or model changes:** build the named targets you will run, such as `ninfer-serve`,
  `ninfer_bench` or one test executable (`ninja -C <tree> ninfer_<area>_test`). Never build `all` or
  `ninfer_tests` for these. Build the aggregate only when the change touches code that everything
  links (`src/core`, `include/ninfer/types.h`, shared op routing, CMake), or for a release.
- **Put the build tree inside your worktree**, as `<worktree>/build-<name>`. ccache shares entries
  between worktrees only when the build tree sits at the same place relative to the sources
  (`cmake/CompilerCache.cmake`). A tree elsewhere, such as `/model-data/scratch/...`, misses every
  CUDA object.
- **Let CMake set the compiler launcher.** Don't pass `-DCMAKE_*_COMPILER_LAUNCHER=ccache`: a bare
  launcher bypasses the settings that make worktrees share. `scripts/v100_build.sh` leaves it to
  CMake.
- **Check the cache after the first build of a new worktree.** Run `ccache -s`: a tree at a commit
  that was already built should mostly hit. A run of CUDA misses means something above is wrong;
  fix that rather than waiting it out.
- **Don't compile while another agent is timing on the GPU.** Check `ps` for `ninfer-serve` or
  `ninfer_bench` first: a big compile raises the load and skews their numbers.

## Waiting on a background job

Wait on the job's **PID**, not on a pattern match against its command line:

```bash
some_long_job & pid=$!                         # or read the PID from `ps`
while kill -0 "$pid" 2>/dev/null; do sleep 20; done
```

**Never** write `while pgrep -f "<string>"` or `until ! pgrep -f "<string>"`. The waiting shell's
own command line contains `<string>`, so `pgrep -f` always finds at least the loop itself, and the
loop never exits. On 2026-11-24, nine such loops were left behind on the V100 host, the oldest after
19 hours. Each finished agent left one or more polling forever, and they had to be killed by hand.

- When the harness can tell you a backgrounded command has finished (a background task that
  notifies on exit), use that rather than a polling loop.
- When you need a pattern because there's no PID, match the program, not the whole command line:
  `pgrep -x ninfer-serve`. Or exclude your own process tree.
- Give every polling loop an upper bound (`timeout 3600 bash -c '...'`, or a counted `for` loop), so
  a mistake ends on its own.
- Before you hand back, check that no loop or sampler of yours is still running: `ps -o pid,etime,cmd
  --ppid <your shell>`, or look for your own commands in `ps`.

## GPU lock (one RTX 3090, many sessions)

Several sessions and agents often work on this machine at once, and they share a single RTX 3090.
Two processes on the card at the same time spoil every measurement and can fail for lack of VRAM. They
coordinate through one lock file:

```
%TEMP%\ninfer-gpu.lock        (C:\Users\Ash\AppData\Local\Temp\ninfer-gpu.lock)
```

The lock is **an open file handle with no sharing**, not the file's existence. The file stays on
disk permanently. Whoever holds a handle opened with `FileShare.None` owns the GPU, and every other
open fails with a sharing violation until that handle is closed. Windows closes the handle when its
process exits, so a crashed or killed holder can never leave the lock stuck.

### What needs the lock

Hold it for **any process that touches CUDA**:

- `ninfer-serve` / `run.bat` server starts, for the server's whole lifetime;
- benches, perplexity runs, `nsys`/`ncu` profiling, and Strata or any other GPU program;
- `ctest` runs of **GPU tests**. Run the suite through `tools/run-tests.ps1`, which does the right
  thing: tests labelled `cpu` run with the GPU hidden and no lock, and each test labelled `gpu`
  takes the lock for that one test only (see "Tests" below).

These do **not** need it: compiling and linking (`cmake --build`), file copies, conversions that run
only on the CPU, reading logs, and the `cpu`-labelled tests (`tools/run-tests.ps1 -CpuOnly`).

### Tests: cpu or gpu

Every test carries one of two labels, set in `tests/cmake/TestGpuClasses.cmake`:

- **`cpu`**: never touches the card. CTest sets `CUDA_VISIBLE_DEVICES=-1` on the test itself, so it
  cannot reach the GPU and needs no lock. If a `cpu` test turns out to need the card it fails
  loudly instead of quietly competing with someone's benchmark.
- **`gpu`**: everything else. A new test is `gpu` until it is added to the CPU-only list, so the
  default is the safe one. To add one, run it with the GPU hidden
  (`$env:CUDA_VISIBLE_DEVICES='-1'; ctest -R <name>`) and confirm it passes **without printing a
  skip**: exit code 0 is not enough (`ninfer_multi_gpu_test` prints `SKIP: no usable CUDA device`
  and still exits 0).

A `cpu` test that links no kernel (not `ninfer_ops`, nor the model, engine or serve libraries) also
goes on `tests/cmake/HostOnlyTests.cmake`. CI builds those without `nvcc` and runs them on every
pull request (the `host-tests` job in `.github/workflows/host-checks.yml`; see "Host-only builds"
in `docs/maintainer/build-system.md`).

```powershell
pwsh tools/run-tests.ps1                      # everything: cpu tests in parallel, then gpu tests
pwsh tools/run-tests.ps1 -CpuOnly             # seconds, no GPU, no lock
pwsh tools/run-tests.ps1 -GpuOnly -Regex kv   # only the gpu tests that match
```

**Run the tests your change needs, not the whole suite.** The full GPU suite takes tens of minutes
on a card that other sessions are waiting for, and most changes cannot affect most of it.

- **Always fine:** `-CpuOnly`. It takes seconds and needs no lock.
- **Then the GPU tests for the area you touched**, picked by name (`ctest -N -R <area>` lists
  them; names follow the code: `ninfer_qwen3_5_*`, `ninfer_kv_cache_*`, `ninfer_host_*`, ...):
  `pwsh tools/run-tests.ps1 -GpuOnly -Regex '<area>'`.
- **The full GPU suite only when it is actually necessary:** a change to code everything links
  (`src/core`, `include/ninfer/types.h`, the shared ops or kernel routing, the CMake or test
  infrastructure), a release, or when the user asks for it. Even then use the runner, never a bare
  `ctest`, and leave out the very slow tests you cannot have affected (`AGENTS.md` lists them;
  `-Exclude 'softmax_attention|kv_materialize'`).
- **A benchmark or serving session runs no tests beyond the ones its own change touches.**
- Say in the PR which tests ran and why the rest did not, so a reviewer can tell a deliberate
  narrow run from a forgotten one.

**Never hold the lock across a whole suite.** The runner takes it per `gpu` test so other waiters
get a turn between tests. A single `ctest` over the full suite under one hold kept the lock for
over 40 minutes (2026-10-13) while GPU utilisation sat between 6% and 27% and other sessions
waited. The slowest tests (`ninfer_softmax_attention_*`, `ninfer_context_kv_materialize_test`,
`AGENTS.md` has their times) still hold the lock for their whole run. If a test spends most of that
on the CPU, holding the lock only around its kernel launches would free the card for others.

### How to take it (PowerShell)

Acquire it, run the GPU work, and release it, all inside **one** PowerShell invocation or script.
Shell state does not survive between tool calls, so a handle opened in one call is gone, and the lock
released, by the next.

`tools/gpu-lock.ps1` has the protocol below as functions, so a script does not copy it:

```powershell
. tools/gpu-lock.ps1
Invoke-WithGpuLock { ... GPU work ... }       # waits (polling every 0.5 s), runs, always releases
```

The hand-written form, for a script that cannot dot-source the helper:

```powershell
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
while (-not $lock) {
  try   { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
  catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }   # someone else holds it
}
try {
  # ... GPU work: bench, ctest, server run ...
} finally {
  $lock.Dispose()                                              # release promptly
}
```

- Waiting can take hours while another agent benchmarks. Put the loop in a script and run it with
  `run_in_background`, rather than polling from the conversation.
- Poll quickly, about every 0.5 s, as above. The lock has no queue: when it is released, the first
  waiter to retry gets it. A waiter polling every 15 s lost to faster pollers again and again and
  waited hours for a few minutes of tests.
- Keep each A/B (ABBA) set inside one hold, so another agent can't run between its halves.
  Release between phases so others can get a turn.
- For a server, keep the handle open until the server process exits, not just until it starts.

### Mistakes that have happened

- **Creating or deleting the file as the lock** (`FileMode.CreateNew`, `New-Item`, `Remove-Item`).
  Other sessions treat a present file as "free" and a missing file as "taken", or the reverse, so
  one agent stalled for about 5 hours waiting on a lock nobody held. Only ever open the existing
  file with `OpenOrCreate` and `FileShare.None`. Never create it fresh, never delete it.
- **Taking the lock in one tool call and running the GPU work in the next.** The handle closed in
  between, so the work ran unlocked.
- **Running `ctest` without the lock** because the regex looked CPU-only. See above. Use
  `tools/run-tests.ps1`: it knows which tests are which, so you do not have to guess.
- **Holding the lock for a whole `ctest` run.** Everyone else waited the length of the suite for a
  GPU that was mostly idle. Take it per GPU test.
- **Running the full suite when a targeted run would do.** One unfiltered `ctest` (no `-R`) held
  the lock for over 40 minutes on 2026-10-13, so the card was unavailable to every other session.
  Run the tests for the area you changed.

### Checking whether the GPU is really in use

If a lock wait feels long, check whether the holder is actually working before suspecting a bug:

```powershell
try { $h = [IO.File]::Open("$env:TEMP\ninfer-gpu.lock", 'OpenOrCreate', 'ReadWrite', 'None'); $h.Dispose(); 'lock free' } catch { 'lock held' }
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
Get-Process ninfer*, nsys* -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU
```

- **Working:** 250-310 W and high utilization means the holder is busy. Flash-Next runs first spend
  about a minute near 40 W, reading the model into RAM.
- **Stuck:** a held lock with an idle GPU and no `ninfer*` process means something is wrong. Find it
  by command line (`Get-CimInstance Win32_Process`).

### The user's own processes

The user's own `ninfer-serve`, games and desktop apps do **not** take the lock. If the GPU is busy
while the lock is free, wait. Don't measure on a busy card, and don't stop the user's processes,
WSL or Docker. Report it if it blocks the work. `AGENTS.md` adds: ask before assuming the device is
free.

The protocol is Windows-only. Share modes don't carry over to WSL `flock`, so GPU work launched from
WSL needs the Windows-side lock held around it.
