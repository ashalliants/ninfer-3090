<#
Runs a list of hosted-workload configurations against the 27B, inside ONE hold of the shared GPU
lock (%TEMP%\ninfer-gpu.lock, see CLAUDE.md), each with its own ninfer-serve process: start the
server with the launcher's tuned MTP flag set plus the config's lanes/Host/store, wait until
/v1/load answers, run tools/bench/run_hosted_workload.py, stop the server (by PID), next config.

  pwsh tools/bench/run_hosted_baseline.ps1 -Plan plan.json -OutRoot runs [-Server exe] [-Model file]

plan.json is a list of objects: name, c (lanes), host (MiB), preset, duration (s), thinking
(on|off), and optionally tenants, think (s), server (exe override), store (directory: adds
--context-store), restart + idle (s): after the run, idle, kill the server (a crash), restart it on
the same store and send one turn per saved conversation into <name>-resume. Keep one plan under
about 45 minutes so other GPU users get a turn. Analyse with tools/bench/hosted_analysis.py.
#>
param(
  [Parameter(Mandatory)] [string] $Plan,
  [Parameter(Mandatory)] [string] $OutRoot,
  [string] $Server = (Join-Path $PSScriptRoot '..\..\build-ninja\apps\ninfer-serve.exe'),
  [string] $Model  = (Join-Path $PSScriptRoot '..\..\models\qwen3_8_27b.ninfer'),
  [int]    $Port   = 8093
)
$ErrorActionPreference = 'Stop'
$Stage = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
New-Item -ItemType Directory -Force $OutRoot | Out-Null
$runs = Get-Content $Plan -Raw | ConvertFrom-Json

function Log($m) { $line = "$(Get-Date -Format 'HH:mm:ss') $m"; Write-Host $line; Add-Content "$OutRoot\phase.log" $line }

function Start-Server($cfg, $dir) {
  $args = @($Model, '--host', '127.0.0.1', '--port', $Port,
    '--max-concurrency', $cfg.c, '--max-context', 262144, '--kv-capacity', 262144,
    '--kv-dtype', 'rk4v4', '--spec', 'mtp', '--draft-tokens', 3, '--lm-head-draft',
    '--gdn-state-fp16', '--prefill-cublas', '--prefill-chunk', 2048,
    '--max-pending-requests', 16, '--pending-timeout-ms', 600000,
    '--vision', '--vision-residency', 'overlay', '--host-context-mib', $cfg.host,
    '--fit-context', 32768, '--min-p', 0.03, '--presence-penalty', 0.5,
    '--request-log-jsonl', "$dir\server-requests.jsonl")
  if ($cfg.store) { $args += @('--context-store', $cfg.store) }
  $exe = if ($cfg.server) { $cfg.server } else { $Server }
  $p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -NoNewWindow `
         -RedirectStandardOutput "$dir\server.out" -RedirectStandardError "$dir\server.err"
  Log "server pid $($p.Id) c=$($cfg.c) host=$($cfg.host) store=$($cfg.store)"
  $deadline = (Get-Date).AddMinutes(10)
  while ((Get-Date) -lt $deadline) {
    if ($p.HasExited) { throw "server exited with $($p.ExitCode) during startup" }
    try { $r = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/v1/load" -TimeoutSec 5
          if ($r.StatusCode -eq 200) { Log "server ready"; return $p } } catch {}
    Start-Sleep -Seconds 2
  }
  throw "server not ready in 10 min"
}

function Stop-Server($p) {
  if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit(60000) | Out-Null }
  Log "server $($p.Id) stopped"
}

function Run-Gen($cfg, $dir, [string[]]$extra) {
  $gargs = @("$Stage\tools\bench\run_hosted_workload.py", '--base-url', "http://127.0.0.1:$Port",
    '--preset', $cfg.preset, '--duration-s', $cfg.duration, '--thinking', $cfg.thinking,
    '--request-log', "$dir\server-requests.jsonl", '--out', $dir) + $extra
  if ($cfg.tenants) { $gargs += @('--tenants', $cfg.tenants) }
  if ($cfg.think) { $gargs += @('--think-s', $cfg.think) }
  Push-Location $Stage
  try { & python @gargs 2> "$dir\client.err" | Tee-Object "$dir\client.out" } finally { Pop-Location }
  Log "generator exit $LASTEXITCODE for $($cfg.name)"
}

$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
Log "waiting for GPU lock"
while (-not $lock) {
  try { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
  catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }
}
Log "lock acquired"
try {
  foreach ($cfg in $runs) {
    $dir = Join-Path $OutRoot $cfg.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    $freeGiB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
    Log "$($cfg.name): free RAM $([math]::Round($freeGiB,1)) GiB, gpu $gpu"
    if ($freeGiB -lt ($cfg.host / 1024 + 6)) { Log "SKIP $($cfg.name): not enough free RAM"; continue }
    if ($cfg.store) { New-Item -ItemType Directory -Force $cfg.store | Out-Null }
    $p = $null
    try {
      $p = Start-Server $cfg $dir
      if ($cfg.restart) {
        Run-Gen $cfg $dir @('--save-state')
        Log "idle $($cfg.idle) s so the store writes idle conversations"
        Start-Sleep -Seconds $cfg.idle
        try { (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/metrics").Content | Set-Content "$dir\metrics-before-kill.txt" } catch {}
        Stop-Server $p   # TerminateProcess: a crash, so only idle-written sessions survive
        $rdir = "$dir-resume"; New-Item -ItemType Directory -Force $rdir | Out-Null
        $p = Start-Server $cfg $rdir
        Run-Gen $cfg $rdir @('--resume-state', "$dir\state.json", '--resume-once')
      } else {
        Run-Gen $cfg $dir @()
      }
    } catch { Log "ERROR $($cfg.name): $_" }
    finally { Stop-Server $p }
  }
} finally {
  $lock.Dispose()
  Log "lock released"
}
