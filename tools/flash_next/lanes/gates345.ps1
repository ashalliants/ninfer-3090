# Lanes PR gates 3 and 5 under ONE hold of the shared GPU lock: the 4K/32K/128K alternation at two
# lanes (gate 3), the Flash-Next cache gates, then ctest for qwen4_exp (gate 5).
#
#   pwsh tools\flash_next\lanes\gates345.ps1 -Log C:\tmp\gates345.log
param(
  [Parameter(Mandatory = $true)][string]$Log,
  [string]$Model = 'C:\ninfer-fork\ninfer-3090\models\qwen4exp-iq2xs-mtp-vision.ninfer'
)
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$build = Join-Path $root 'build-ninja'
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
$exitCode = 0
$failed = @()
$t0 = Get-Date
while (-not $lock) {
  try   { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
  catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }
}
function Log($s) { $s | Out-File $Log -Append -Encoding utf8 }
function Done($name) {
  $code = $LASTEXITCODE
  Log "exit $code"
  if ($code -ne 0) { $script:failed += "$name (exit $code)" }
}
try {
  "lock acquired after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds | Out-File $Log -Encoding utf8
  $env:NINFER_TEST_QWEN4EXP_ARTIFACT = $Model
  $quiet = '^\[qwen3.8-flash-next\] (expert cache:|PCIe|VRAM|prefill lends|wide prefill|mtp:)'
  Log '=== gate 3: 4K/32K/128K alternation, two lanes, MTP'
  & "$build\tests\ninfer_qwen4_exp_lanes_real_test.exe" --bench --concurrency 2 --mtp 3 --generate 64 --max-context 135168 `
      4096 130000 4096 4096 32000 130000 4096 32000 4096 130000 2>&1 | Select-String -NotMatch $quiet | ForEach-Object { Log $_.Line }
  Done 'gate 3'
  $runs = @(
    @('cache, plain decode', 'ninfer_qwen4_exp_engine_real_test', '--cache', '--generate', '32'),
    @('cache, MTP 3', 'ninfer_qwen4_exp_engine_real_test', '--cache', '--generate', '32', '--mtp', '3'),
    @('cache, eviction at 512 MiB', 'ninfer_qwen4_exp_engine_real_test', '--cache', '--evict', '--host-mib', '512', '--generate', '32'),
    @('vision cache (MTP)', 'ninfer_qwen4_exp_vision_engine_real_test', '--cache')
  )
  foreach ($run in $runs) {
    Log "=== $($run[0])"
    & "$build\tests\ninfer_tests.exe" $run[1..($run.Count - 1)] 2>&1 | Select-String -NotMatch $quiet | ForEach-Object { Log $_.Line }
    Done $run[0]
  }
  Log '=== gate 5: ctest -R qwen4_exp'
  cmd /c "`"$vcvars`" >nul 2>&1 && cd /d `"$build`" && ctest -R qwen4_exp --output-on-failure --timeout 3000" 2>&1 | ForEach-Object { Log $_ }
  Done 'gate 5 ctest'
  if ($failed.Count -gt 0) {
    Log "FAILED: $($failed -join '; ')"
    $exitCode = 1
  } else {
    Log 'ALL DONE'
  }
} finally { $lock.Dispose() }
exit $exitCode
