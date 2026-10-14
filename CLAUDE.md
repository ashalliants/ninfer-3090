# CLAUDE.md

`AGENTS.md` holds the project rules and the Windows build recipe. This file adds what Claude sessions
on this machine need to share the GPU safely.

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
