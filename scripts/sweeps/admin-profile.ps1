# Everything in this repository that needs an elevated shell, in one run.
#
# THIS MUST BE RUN AS ADMINISTRATOR. Open Windows Terminal or PowerShell with "Run as
# administrator" and run it from the repository root:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\admin-profile.ps1
#
# It needs admin for exactly two reasons, both verified on this box on 2026-09-09:
#
#   * ncu fails with ERR_NVGPUCTRPERM as a normal user. GPU performance counters are
#     administrator-only by default on Windows. That permission is satisfied by EITHER running
#     elevated OR setting HKLM\...\NvTweak\RmProfilingAdminOnly = 0 and rebooting. Running
#     elevated is strictly better: nothing persists, and no reboot.
#   * nvidia-smi -pl and -lgc are refused with "Insufficient Permissions" as a normal user.
#
# It changes two pieces of GPU state -- the power limit and the clock lock -- and restores both in
# a finally block, including on Ctrl+C. Nothing it does survives the script.
#
# Results land under profiles\admin\ as text files. Nothing here needs interpreting live; the
# point is to capture what an unelevated session cannot.
[CmdletBinding()]
param(
  [string]$ModelDir = $(if ($env:NINFER_MODEL_DIR) { $env:NINFER_MODEL_DIR } else { 'C:\Ninefer-3090\models' }),
  [string]$Ncu      = $(if ($env:NINFER_NCU) { $env:NINFER_NCU } else { 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\ncu.bat' }),
  [switch]$SkipPower
)
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Error 'Not elevated. Re-run this from an administrator shell -- everything here needs it.'
  exit 1
}

$out = 'profiles\admin'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$bench = '.\build-ninja\bench\ninfer_bench.exe'
$moe   = "$ModelDir\qwen3_6_35b_a3b.ninfer"
$dense = "$ModelDir\qwen3_8_27b.ninfer"
foreach ($p in @($bench, $moe, $dense)) {
  if (-not (Test-Path $p)) { Write-Error "missing: $p"; exit 1 }
}
if (-not (Test-Path $Ncu)) { Write-Error "ncu not found at $Ncu; set NINFER_NCU"; exit 1 }

# Remember what to put back. power.limit is the enforced ceiling; default_limit is the factory one.
$originalLimit = (nvidia-smi --query-gpu=power.limit --format=csv,noheader | Select-Object -First 1)
$originalLimit = [double]($originalLimit -replace '[^0-9.]', '')
"original power limit: $originalLimit W"
$clocksLocked = $false
$powerChanged = $false

try {
  # ---------------------------------------------------------------- 1. the MoE expert gather
  #
  # TODO section 2c: sparse_moe_d3/d4 reach 40-45% of the card's read bandwidth where contiguous
  # weight kernels on the same decode step reach 78-82%, and the four sparse_moe stages are 38% of
  # decode busy time on the 35B. Three candidate causes -- address divergence from gathering eight
  # scattered experts of 256, L2 behaviour, or too little work per CTA to cover latency -- and
  # nsys cannot distinguish them. These sections can.
  #
  # Clocks are locked first so the counters describe one clock state rather than an average of the
  # card ramping. --graph-profiling defaults to node, which matters because decode replays a
  # captured CUDA graph and the default in older tools records a replay as one opaque entity.
  "locking clocks for the counter run"
  nvidia-smi -lgc 1500 2>&1 | Out-String | Write-Host
  if ($LASTEXITCODE -eq 0) { $clocksLocked = $true }

  "== 1/3 ncu: MoE expert gather (35B, int8, decode) -> $out\moe_gather.txt"
  & $Ncu --target-processes all --graph-profiling node `
      -k 'regex:sparse_moe' --launch-skip 64 --launch-count 12 `
      --section SpeedOfLight --section MemoryWorkloadAnalysis `
      --section MemoryWorkloadAnalysis_Tables --section Occupancy --section LaunchStats `
      --section SchedulerStats --section WarpStateStats `
      $bench --weights $moe --kv-dtype int8 --max-ctx 8192 -n 64 -r 1 --warmup 1 `
      > "$out\moe_gather.txt" 2>&1
  "  ncu exit $LASTEXITCODE"
  if (Select-String -Path "$out\moe_gather.txt" -Pattern 'ERR_NVGPUCTRPERM' -Quiet) {
    "  STILL PERMISSION-DENIED -- the shell is elevated but the driver refused; the registry"
    "  route (RmProfilingAdminOnly=0 plus reboot) is then the only option."
  }

  # A contiguous kernel from the same step, as the reference the percentages above are against.
  # Without it the MoE numbers have no local baseline and have to be compared across runs.
  "== 2/3 ncu: contiguous reference kernels (27B, int8, decode) -> $out\contiguous_ref.txt"
  & $Ncu --target-processes all --graph-profiling node `
      -k 'regex:q5_rowsplit_gemv|q4_linear_swiglu_gemv_pair' --launch-skip 64 --launch-count 8 `
      --section SpeedOfLight --section MemoryWorkloadAnalysis `
      --section MemoryWorkloadAnalysis_Tables --section Occupancy --section LaunchStats `
      $bench --weights $dense --kv-dtype int8 --max-ctx 8192 -n 64 -r 1 --warmup 1 `
      > "$out\contiguous_ref.txt" 2>&1
  "  ncu exit $LASTEXITCODE"

  # ---------------------------------------------------------------- 2. what the power cap costs
  #
  # TODO section 3: the 315 W cap binds in essentially every busy sample, but memory clock never
  # leaves 9,501 MHz, so the bandwidth figures are already unthrottled and the prediction is that
  # the missing 35 W buys very little. This is the measurement that settles it. Clocks are released
  # first -- a locked clock would hide exactly the effect being looked for.
  if (-not $SkipPower) {
    if ($clocksLocked) { nvidia-smi -rgc 2>&1 | Out-Null; $clocksLocked = $false }
    "== 3/3 power limit 350 W -> $out\power_350.txt"
    nvidia-smi -pl 350 2>&1 | Out-String | Write-Host
    if ($LASTEXITCODE -eq 0) {
      $powerChanged = $true
      powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\power-and-clocks.ps1 `
          > "$out\power_350.txt" 2>&1
      Get-Content "$out\power_350.txt" | Select-String '^==|^  ' | ForEach-Object { "  $_" }
    } else {
      "  -pl 350 refused even elevated; skipping"
    }
  }
}
finally {
  # Restore unconditionally. A left-behind clock lock or power limit silently changes every
  # measurement taken afterwards, which is worse than not having run this at all.
  if ($clocksLocked) { "restoring clocks"; nvidia-smi -rgc 2>&1 | Out-Null }
  if ($powerChanged) {
    "restoring power limit to $originalLimit W"
    nvidia-smi -pl $originalLimit 2>&1 | Out-Null
  }
  nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv | ForEach-Object { "  $_" }
}

""
"== done. Wrote:"
Get-ChildItem $out -File | ForEach-Object { "   {0,10:N0} bytes  {1}" -f $_.Length, $_.FullName }
""
"The two ncu files are the ones that matter; they are long, and are meant to be read by whoever"
"asked for them rather than skimmed here."
