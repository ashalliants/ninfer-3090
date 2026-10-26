<#
Verifies the Basalt-learnings options on the shared Windows box: runs tools/bench/basalt_gpu_verify.py
one step at a time and holds the shared GPU lock (tools/gpu-lock.ps1, CLAUDE.md "GPU lock") around
each step that uses the card, releasing it between steps so other sessions get a turn.

  pwsh tools/bench/basalt_gpu_verify.ps1 --build-dir build-ninja --dry-run
  pwsh tools/bench/basalt_gpu_verify.ps1 --build-dir build-ninja `
      --flash-next K:\ninfer-models\qwen4exp\qwen4exp-iq2xs-mtp.ninfer `
      --qwen27b C:\ninfer-fork\ninfer-3090\models\qwen3_8_27b.ninfer `
      --kld-base-flash-next K:\kld\flash-next-q8_0-512.kld --master-cli build-ninja\apps\ninfer-master.exe

Every argument goes to basalt_gpu_verify.py unchanged; `python tools/bench/basalt_gpu_verify.py --help`
lists them. $env:NINFER_PYTHON picks the interpreter (default `python`). The results directory is
resolved once, before the first step, and passed to every call, so a run that crosses midnight stays
in one directory; add --resume to continue a run (it leaves out the steps already PASS, MEASURED or
REVIEW, and without --out picks the newest run).

Locking, per the step's lock column (`--action steps` prints it):
  hold      this script holds the lock for the whole step: one ABBA set, one KLD run, one server
            start and its probes. An A/B's two arms therefore never have another job between them.
  per-test  the step runs tools/run-tests.ps1, which takes the lock for each GPU test on its own.
  none      no GPU (the packager, pytest, the summary), or a step that will SKIP for a missing input,
            which therefore does not wait for the lock.
A wait can take hours while another session benchmarks: start this with run_in_background rather
than watching it. Results and summary.md go to --out (default profiles\bench\basalt-<date>\).
#>
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\gpu-lock.ps1')
$python = if ($env:NINFER_PYTHON) { $env:NINFER_PYTHON } else { 'python' }
$core = Join-Path $PSScriptRoot 'basalt_gpu_verify.py'
$pass = @($args)

if ($pass -contains '--dry-run' -or $pass -contains '--help' -or $pass -contains '-h') {
  & $python $core @pass
  exit $LASTEXITCODE
}

$out = & $python $core @pass --action out
if ($LASTEXITCODE -ne 0 -or -not $out) { throw 'basalt_gpu_verify.py --action out failed; run it with --dry-run to see why.' }
$pass += @('--out', "$out")

$steps = @(& $python $core @pass --action steps)
if ($LASTEXITCODE -ne 0) { throw 'basalt_gpu_verify.py --action steps failed; run it with --dry-run to see why.' }
$failed = @()
foreach ($line in $steps) {
  $id, $lock, $title = $line -split "`t", 3
  $script:code = 0
  if ($lock -eq 'hold') {
    $script:waited = [Diagnostics.Stopwatch]::StartNew()
    Invoke-WithGpuLock -OnWait { Write-Host "   waiting for the GPU lock before $id ..." } -Script {
      $script:waited.Stop()
      & $python $core @pass --action run --step $id
      $script:code = $LASTEXITCODE
    }
    Write-Host ("   {0}: waited {1:N0} s for the lock" -f $id, $script:waited.Elapsed.TotalSeconds)
  } else {
    & $python $core @pass --action run --step $id
    $script:code = $LASTEXITCODE
  }
  if ($script:code -ne 0) { $failed += $id }
}

& $python $core @pass --action summary
if ($failed) {
  Write-Host "Steps that failed: $($failed -join ', ')"
  exit 1
}
