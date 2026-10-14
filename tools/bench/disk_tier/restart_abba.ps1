# Seed a store once, then continue each conversation after a restart, ABBA, inside one lock hold.
param([string]$Tag, [string]$BinA, [string]$BinB, [string]$SeedBin, [int]$HostMib = 1024,
      [string]$Order = 'ABBA', [string]$Sizes = '1100,1900,3000,4600,7600,12000')
$ErrorActionPreference = 'Stop'
$s = $PSScriptRoot
$runs = if ($env:NINFER_DISK_TIER_RUNS) { $env:NINFER_DISK_TIER_RUNS } else { Join-Path $env:TEMP "ninfer-disk-tier-runs" }
$model = if ($env:NINFER_MODEL) { $env:NINFER_MODEL } else { "C:\ninfer-fork\ninfer-3090\models\qwen3_8_27b.ninfer" }
$out = "$runs\$Tag"; New-Item -ItemType Directory -Force $out | Out-Null
$port = 18200
$seedStore = "$env:TEMP\ninfer-disk-runs\$Tag-seed"
function Start-Server($bin, $store, $hostMib, $name) {
  $sargs = @($model, '--host', '127.0.0.1', '--port', "$port", '--max-concurrency', '4',
    '--max-context', '188416', '--kv-capacity', '188416', '--kv-dtype', 'rk4v4',
    '--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft', '--gdn-state-fp16',
    '--prefill-cublas', '--prefill-chunk', '4096', '--host-context-mib', "$hostMib", '--fit-context', '32768',
    '--min-p', '0.03', '--presence-penalty', '0.5', '--context-store', $store, '--context-store-idle-seconds', '2')
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
function Metric($name) {
  $m = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/metrics").Content -split "`n" | Where-Object { $_ -match "^$name " }
  if ($m) { return [double]($m -split ' ')[1] } else { return 0 }
}
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
while (-not $lock) { try { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') } catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 } }
"lock acquired at $(Get-Date -Format o)" | Tee-Object -FilePath "$out\lock.txt"
try {
  if (Test-Path $seedStore) { Remove-Item -Recurse -Force $seedStore }
  $p = Start-Server $SeedBin $seedStore 8192 'seed'
  try {
    & python -I "$s\restart_probe.py" seed --base-url "http://127.0.0.1:$port" --sizes $Sizes --state "$out\state.json" 2>&1 | Tee-Object -FilePath "$out\seed.txt"
    if ($LASTEXITCODE -ne 0) { throw "seed probe exited $LASTEXITCODE (incomplete run, see $out\seed.txt)" }
    $n = ($Sizes -split ',').Count
    for ($t = 0; $t -lt 120; $t++) { if ((Metric 'ninfer_context_store_writes_total') -ge $n) { break }; Start-Sleep -Seconds 1 }
    Start-Sleep -Seconds 3
    "seed writes=$(Metric 'ninfer_context_store_writes_total') images=$(Metric 'ninfer_context_store_images')" | Tee-Object -Append -FilePath "$out\lock.txt"
  } finally { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
  $i = 0
  foreach ($arm in $Order.ToCharArray()) {
    $i++
    $bin = if ($arm -eq 'A') { $BinA } else { $BinB }
    $store = "$env:TEMP\ninfer-disk-runs\$Tag-$i$arm"
    if (Test-Path $store) { Remove-Item -Recurse -Force $store }
    xcopy /J /E /I /Q $seedStore $store | Out-Null
    $p = Start-Server $bin $store $HostMib "$i$arm"
    try {
      & python -I "$s\restart_probe.py" continue --base-url "http://127.0.0.1:$port" --state "$out\state.json" --out "$out\cont-$i$arm.json" 2>&1 | Tee-Object -FilePath "$out\cont-$i$arm.txt"
      if ($LASTEXITCODE -ne 0) { throw "continue probe $i$arm exited $LASTEXITCODE (incomplete run, see $out\cont-$i$arm.txt)" }
      "run $i$arm restored=$(Metric 'ninfer_context_store_restored_sessions') hydrations=$(Metric 'ninfer_context_store_hydrations_total') hydrated_tokens=$(Metric 'ninfer_context_store_hydrated_tokens_total') failures=$(Metric 'ninfer_context_store_hydration_failures_total') hyd_s=$(Metric 'ninfer_context_store_hydration_seconds_total') rate=$(Metric 'ninfer_context_store_read_bytes_per_second')" | Tee-Object -Append -FilePath "$out\lock.txt"
    } finally { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
    Remove-Item -Recurse -Force $store
  }
  Remove-Item -Recurse -Force $seedStore
} finally {
  $lock.Dispose()
  "lock released at $(Get-Date -Format o)" | Tee-Object -Append -FilePath "$out\lock.txt"
}
