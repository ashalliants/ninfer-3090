# ABBA of two ninfer-serve builds on the hosted rotation, inside ONE GPU-lock hold.
param(
  [string]$Tag,
  [string]$BinA, [string]$BinB, [string]$BinC = '',
  [int]$Concurrency = 4,
  [int]$HostMib = 2048,
  [string]$Order = 'ABBA',
  [int]$Requests = 80,
  [int]$Tenants = 24,
  [switch]$NoStore,
  [string]$ExtraArgs = '',
  [string]$DriverArgs = '',
  [string]$Spec = 'dflash2'
)
$ErrorActionPreference = 'Stop'
$s = $PSScriptRoot
$runs = if ($env:NINFER_DISK_TIER_RUNS) { $env:NINFER_DISK_TIER_RUNS } else { Join-Path $env:TEMP "ninfer-disk-tier-runs" }
$model = if ($env:NINFER_MODEL) { $env:NINFER_MODEL } else { "C:\ninfer-fork\ninfer-3090\models\qwen3_8_27b.ninfer" }
$out = "$runs\$Tag"; New-Item -ItemType Directory -Force $out | Out-Null
$port = 18200
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
$waitStart = Get-Date
while (-not $lock) {
  try { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
  catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }
}
"lock acquired after $([int]((Get-Date) - $waitStart).TotalSeconds) s at $(Get-Date -Format o)" | Tee-Object -FilePath "$out\lock.txt"
try {
  $i = 0
  foreach ($arm in $Order.ToCharArray()) {
    $i++
    $bin = switch ($arm) { 'A' { $BinA } 'B' { $BinB } 'C' { $BinC } }
    $store = "$env:TEMP\ninfer-disk-runs\$Tag-$i$arm"
    if (Test-Path $store) { Remove-Item -Recurse -Force $store }
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $smi = nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
    "run $i$arm free_ram_gib=$([math]::Round($free,1)) gpu=$smi" | Tee-Object -Append -FilePath "$out\lock.txt"
    $sargs = @($model, '--host', '127.0.0.1', '--port', "$port", '--max-concurrency', "$Concurrency",
      '--kv-dtype', 'rk4v4', '--lm-head-draft', '--gdn-state-fp16', '--prefill-cublas', '--max-pending-requests', '16',
      '--pending-timeout-ms', '600000', '--host-context-mib', "$HostMib", '--fit-context', '32768',
      '--min-p', '0.03', '--presence-penalty', '0.5')
    if ($Spec -eq 'mtp') { $sargs += @('--max-context', '262144', '--kv-capacity', '262144', '--spec', 'mtp', '--draft-tokens', '3', '--prefill-chunk', '2048') }
    else { $sargs += @('--max-context', '188416', '--kv-capacity', '188416', '--spec', 'dflash2', '--draft-tokens', '7', '--prefill-chunk', '4096') }
    if (-not $NoStore) { $sargs += @('--context-store', $store) }
    if ($ExtraArgs) { $sargs += $ExtraArgs.Split(' ') }
  for ($w = 0; $w -lt 90; $w++) { $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1); if ($used -lt 2500) { break }; Start-Sleep -Seconds 1 }
    $p = Start-Process -FilePath "$bin\ninfer-serve.exe" -ArgumentList $sargs -PassThru -NoNewWindow `
      -RedirectStandardOutput "$out\server-$i$arm.log" -RedirectStandardError "$out\server-$i$arm.err"
    try {
      $ready = $false
      for ($t = 0; $t -lt 600 -and -not $p.HasExited; $t++) {
        try { $r = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/v1/models" -TimeoutSec 2; if ($r.StatusCode -eq 200) { $ready = $true; break } } catch {}
        Start-Sleep -Seconds 1
      }
      if (-not $ready) { throw "server $i$arm did not become ready" }
      $dargs = @("$s\hosted_rotation.py", '--base-url', "http://127.0.0.1:$port", '--workers', "$Concurrency",
        '--requests', "$Requests", '--tenants', "$Tenants", '--out', "$out\result-$i$arm.json")
      if ($DriverArgs) { $dargs += $DriverArgs.Split(' ') }
      & python -I @dargs 2> "$out\driver-$i$arm.log" | Out-File "$out\summary-$i$arm.json"
      if ($LASTEXITCODE -ne 0) { throw "driver $i$arm exited $LASTEXITCODE (incomplete run, see $out\driver-$i$arm.log)" }
    } finally {
      if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
    }
    if (Test-Path $store) {
      $bytes = (Get-ChildItem -Recurse -File $store | Measure-Object -Sum Length).Sum
      "store $i$arm bytes_on_disk=$bytes" | Tee-Object -Append -FilePath "$out\lock.txt"
      Remove-Item -Recurse -Force $store
    }
  }
} finally {
  $lock.Dispose()
  "lock released at $(Get-Date -Format o)" | Tee-Object -Append -FilePath "$out\lock.txt"
}
