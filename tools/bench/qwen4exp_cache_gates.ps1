# Runs the Qwen3.8-Flash-Next context-cache real-model gates under the shared GPU lock
# (CLAUDE.md protocol: open %TEMP%\ninfer-gpu.lock with FileShare.None, poll every 0.5 s, hold it
# for the whole run, release in finally; never create or delete the file).
#
#   pwsh tools/bench/qwen4exp_cache_gates.ps1 -Tests build-ninja\tests\ninfer_tests.exe `
#        [-Model C:\ninfer-fork\ninfer-3090\models\qwen4exp-iq2xs-mtp-vision.ninfer]
#
# Gates: resume == cold (plain and MTP 3), eviction under a 512 MiB Host tier, image prefix.
# Each prints "... checks passed" and exits 0 when green. Needs ~40 GiB of available RAM (the
# model pins 33 GiB of experts); only one Flash-Next process may run on the machine at a time.
param(
    [Parameter(Mandatory = $true)][string]$Tests,
    [string]$Model = 'C:\ninfer-fork\ninfer-3090\models\qwen4exp-iq2xs-mtp-vision.ninfer'
)
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
$t0 = Get-Date
while (-not $lock) {
    try { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
    catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }
}
"[gpu-lock] acquired after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds
$failed = 0
try {
    $free = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
    "[gpu-lock] available RAM {0:N1} GiB" -f $free
    if ($free -lt 38) { throw "less than 38 GiB of RAM available" }
    $env:NINFER_TEST_QWEN4EXP_ARTIFACT = $Model
    $quiet = '^\[qwen3.8-flash-next\] (expert cache:|PCIe|VRAM|prefill lends|wide prefill|mtp:)'
    $runs = @(
        @('cache, plain decode', 'ninfer_qwen4_exp_engine_real_test', '--cache', '--generate', '32'),
        @('cache, MTP 3', 'ninfer_qwen4_exp_engine_real_test', '--cache', '--generate', '32', '--mtp', '3'),
        @('cache, eviction at 512 MiB', 'ninfer_qwen4_exp_engine_real_test', '--cache', '--evict', '--host-mib', '512', '--generate', '32'),
        @('vision cache (MTP)', 'ninfer_qwen4_exp_vision_engine_real_test', '--cache')
    )
    foreach ($run in $runs) {
        "=== $($run[0])"
        & $Tests $run[1..($run.Count - 1)] 2>&1 | Select-String -NotMatch $quiet
        "exit $LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { $failed++ }
    }
} finally {
    $lock.Dispose()
    "[gpu-lock] released"
}
if ($failed -ne 0) { "$failed gate(s) failed"; exit 1 }
"all gates passed"
