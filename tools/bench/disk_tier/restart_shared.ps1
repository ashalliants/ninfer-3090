# Restart warm-start ABBA: each arm seeds its own store with the hosted rotation, is killed (a
# crash: no shutdown flush), restarts on that store and serves NEW tenants on the same system
# prompts. One GPU-lock hold.
param([string]$Tag, [string]$BinA, [string]$BinB, [string]$Order = 'ABBA', [int]$HostMib = 2048,
      [int]$SeedRequests = 48, [int]$Requests = 24)
$ErrorActionPreference = 'Stop'
$s = $PSScriptRoot
$runs = if ($env:NINFER_DISK_TIER_RUNS) { $env:NINFER_DISK_TIER_RUNS } else { Join-Path $env:TEMP "ninfer-disk-tier-runs" }
$model = if ($env:NINFER_MODEL) { $env:NINFER_MODEL } else { "C:\ninfer-fork\ninfer-3090\models\qwen3_8_27b.ninfer" }
$out = "$runs\$Tag"; New-Item -ItemType Directory -Force $out | Out-Null
$port = 18200
function Start-Server($bin, $store, $name) {
  $sargs = @($model, '--host', '127.0.0.1', '--port', "$port", '--max-concurrency', '4',
    '--max-context', '188416', '--kv-capacity', '188416', '--kv-dtype', 'rk4v4',
    '--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft', '--gdn-state-fp16',
    '--prefill-cublas', '--prefill-chunk', '4096', '--max-pending-requests', '16',
    '--pending-timeout-ms', '600000', '--host-context-mib', "$HostMib", '--fit-context', '32768',
    '--min-p', '0.03', '--presence-penalty', '0.5', '--context-store', $store)
  for ($w = 0; $w -lt 90; $w++) { $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1); if ($used -lt 2500) { break }; Start-Sleep -Seconds 1 }
  $p = Start-Process -FilePath "$bin\ninfer-serve.exe" -ArgumentList $sargs -PassThru -NoNewWindow `
    -RedirectStandardOutput "$out\server-$name.log" -RedirectStandardError "$out\server-$name.err"
  for ($t = 0; $t -lt 600 -and -not $p.HasExited; $t++) {
    try { $r = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/v1/models" -TimeoutSec 2; if ($r.StatusCode -eq 200) { return $p } } catch {}
    Start-Sleep -Seconds 1
  }
  if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
  throw "server $name did not become ready"
}
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
while (-not $lock) { try { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') } catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 } }
"lock acquired at $(Get-Date -Format o)" | Tee-Object -FilePath "$out\lock.txt"
try {
  $i = 0
  foreach ($arm in $Order.ToCharArray()) {
    $i++
    $bin = if ($arm -eq 'A') { $BinA } else { $BinB }
    $store = "$env:TEMP\ninfer-disk-runs\$Tag-$i$arm"
    if (Test-Path $store) { Remove-Item -Recurse -Force $store }
    $p = Start-Server $bin $store "$i$arm-seed"
    try {
      & python -I "$s\hosted_rotation.py" --base-url "http://127.0.0.1:$port" --workers 4 --requests $SeedRequests --seed 1 --system-seed 1 --out "$out\seed-$i$arm.json" 2> "$out\seed-$i$arm.log" | Out-File "$out\seed-summary-$i$arm.json"
      if ($LASTEXITCODE -ne 0) { throw "seed driver $i$arm exited $LASTEXITCODE (incomplete run, see $out\seed-$i$arm.log)" }
      Start-Sleep -Seconds 5
    } finally { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
    $bytes = (Get-ChildItem -Recurse -File $store | Measure-Object -Sum Length).Sum
    "seed $i$arm store_bytes=$bytes" | Tee-Object -Append -FilePath "$out\lock.txt"
    $p = Start-Server $bin $store "$i$arm-warm"
    try {
      $restored = ((Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/metrics").Content -split "`n" | Where-Object { $_ -match '^ninfer_context_store_restored_sessions ' })
      "warm $i$arm $restored" | Tee-Object -Append -FilePath "$out\lock.txt"
      & python -I "$s\hosted_rotation.py" --base-url "http://127.0.0.1:$port" --workers 4 --requests $Requests --seed 2 --system-seed 1 --out "$out\warm-$i$arm.json" 2> "$out\warm-$i$arm.log" | Out-File "$out\warm-summary-$i$arm.json"
      if ($LASTEXITCODE -ne 0) { throw "warm driver $i$arm exited $LASTEXITCODE (incomplete run, see $out\warm-$i$arm.log)" }
    } finally { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
    Remove-Item -Recurse -Force $store
  }
} finally {
  $lock.Dispose()
  "lock released at $(Get-Date -Format o)" | Tee-Object -Append -FilePath "$out\lock.txt"
}
