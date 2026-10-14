<#
Runs the CTest suite so that nobody waits for the GPU without needing it.

  pwsh tools/run-tests.ps1                    # everything
  pwsh tools/run-tests.ps1 -CpuOnly           # no GPU, no lock: seconds
  pwsh tools/run-tests.ps1 -GpuOnly -Regex kv_cache
  pwsh tools/run-tests.ps1 -GpuOnly -Exclude 'softmax_attention|kv_materialize'   # skip the slow ones
  pwsh tools/run-tests.ps1 -BuildDir C:\path\to\build-ninja -Jobs 8

Prefer a targeted run (-Regex for the area you changed) to the whole suite: see CLAUDE.md.

Every test is labelled `cpu` or `gpu` (tests/cmake/TestGpuClasses.cmake; a test is `gpu` unless it
is on the CPU-only list).
  * `cpu` tests run in parallel with the GPU hidden (CUDA_VISIBLE_DEVICES=-1 is set on the test
    itself), so they cannot touch the card and need no lock. One that wrongly needs it fails
    loudly instead of quietly competing with someone's benchmark.
  * `gpu` tests run one at a time, each inside its own short hold of the shared GPU lock
    (tools/gpu-lock.ps1), so other waiters get a turn between tests instead of waiting for the
    whole suite. Never hold the lock across a whole suite.

Run this from the build tree's environment (see AGENTS.md). Exit status is 1 if any test failed.
#>
param(
  [string] $BuildDir = (Join-Path $PSScriptRoot '..\build-ninja'),
  [string] $Regex,                  # only tests whose name matches (ctest -R)
  [string] $Exclude,                # leave out tests whose name matches (ctest -E)
  [switch] $CpuOnly,
  [switch] $GpuOnly,
  [int]    $Jobs    = 4,            # parallelism of the cpu phase
  [int]    $Timeout = 3600,         # seconds per test
  [string[]] $CtestArgs = @()
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\gpu-lock.ps1"
if ($CpuOnly -and $GpuOnly) { throw 'Choose at most one of -CpuOnly and -GpuOnly.' }
$BuildDir = (Resolve-Path $BuildDir).Path

function Select-Tests([string] $label) {
  $listArgs = @('--test-dir', $BuildDir, '-N', '-L', "^$label`$")
  if ($Regex) { $listArgs += @('-R', $Regex) }
  if ($Exclude) { $listArgs += @('-E', $Exclude) }
  (& ctest @listArgs) | ForEach-Object { if ($_ -match 'Test\s+#\d+:\s+(\S+)') { $Matches[1] } }
}

$cpuTests = @(Select-Tests 'cpu')
$gpuTests = @(Select-Tests 'gpu')
if (-not $cpuTests -and -not $gpuTests) {
  throw "No labelled tests found in $BuildDir. Re-run cmake there once so the cpu/gpu labels exist."
}
$failed = @()

if (-not $GpuOnly -and $cpuTests) {
  Write-Host "== $($cpuTests.Count) cpu tests, GPU hidden, no lock, -j$Jobs"
  $cpuArgs = @('--test-dir', $BuildDir, '-L', '^cpu$', '-j', $Jobs, '--timeout', $Timeout,
               '--output-on-failure') + $CtestArgs
  if ($Regex) { $cpuArgs += @('-R', $Regex) }
  if ($Exclude) { $cpuArgs += @('-E', $Exclude) }
  & ctest @cpuArgs
  if ($LASTEXITCODE -ne 0) { $failed += 'cpu phase' }
}

if (-not $CpuOnly -and $gpuTests) {
  Write-Host "== $($gpuTests.Count) gpu tests, one at a time, each under its own lock hold"
  foreach ($name in $gpuTests) {
    $script:waited = [Diagnostics.Stopwatch]::StartNew()
    $script:held   = [Diagnostics.Stopwatch]::new()
    $script:code   = 0
    Invoke-WithGpuLock -OnWait { Write-Host "   waiting for the GPU lock before $name ..." } -Script {
      $script:waited.Stop()
      $script:held.Start()
      & ctest --test-dir $BuildDir -R "^$name`$" --timeout $Timeout --output-on-failure @CtestArgs
      $script:code = $LASTEXITCODE
      $script:held.Stop()
    }
    Write-Host ("   {0}: waited {1:N1}s, held {2:N1}s, exit {3}" -f $name,
                $script:waited.Elapsed.TotalSeconds, $script:held.Elapsed.TotalSeconds, $script:code)
    if ($script:code -ne 0) { $failed += $name }
  }
}

if ($failed) { Write-Host "FAILED: $($failed -join ', ')"; exit 1 }
Write-Host 'All selected tests passed.'
