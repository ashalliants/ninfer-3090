# Gate 4 of the lanes PR: a lone request's speed. The spike's eight conversations one at a time
# (decode and prefill figures from the server's own log) on master, on this branch at
# --max-concurrency 1 and at 2, in M L1 L2 L2 L1 M order inside ONE hold of the shared GPU lock.
# Each server is started and stopped by its own PID.
#
#   pwsh tools\flash_next\lanes\lone_gate.ps1 -OutDir C:\tmp\lone -MasterRoot <a master checkout with ninfer-serve built>
param(
  [Parameter(Mandatory = $true)][string]$OutDir,
  [Parameter(Mandatory = $true)][string]$MasterRoot,
  [string[]]$Order = @('master:1', 'lanes:1', 'lanes:2', 'lanes:2', 'lanes:1', 'master:1'),
  [string]$Model = 'C:\ninfer-fork\ninfer-3090\models\qwen4exp-iq2xs-mtp-vision.ninfer',
  [int]$Port = 18292
)
$root   = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$client = Join-Path $PSScriptRoot 'burst_client.py'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
$exitCode = 0
$t0 = Get-Date
while (-not $lock) {
  try   { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
  catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }
}
try {
  "lock acquired after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds | Out-File "$OutDir\gate.log" -Encoding utf8
  $n = 0
  $failed = @()
  foreach ($entry in $Order) {
    $n++
    $which, $c = $entry -split ':'
    $server = if ($which -eq 'master') { Join-Path $MasterRoot 'build-ninja\apps\ninfer-serve.exe' } else { Join-Path $root 'build-ninja\apps\ninfer-serve.exe' }
    $tag = "run${n}_${which}_C$c"
    $srvArgs = @($Model, '--host', '127.0.0.1', '--port', $Port, '--max-concurrency', $c, '--max-context', 32768,
                 '--kv-capacity', 32768, '--kv-dtype', 'int8', '--prefill-chunk', 1024, '--spec', 'mtp', '--draft-tokens', 3,
                 '--no-expert-state', '--no-prefix-reuse')
    $p = Start-Process -FilePath $server -ArgumentList $srvArgs -PassThru -WindowStyle Hidden `
         -RedirectStandardOutput "$OutDir\$tag.server.out" -RedirectStandardError "$OutDir\$tag.server.err"
    "$tag server pid $($p.Id)" | Out-File "$OutDir\gate.log" -Append -Encoding utf8
    try {
      $ready = $false
      for ($i = 0; $i -lt 600 -and -not $ready -and -not $p.HasExited; $i++) {
        try { $null = Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3; $ready = $true } catch { Start-Sleep -Seconds 2 }
      }
      if (-not $ready) { "$tag server not ready (exited: $($p.HasExited))" | Out-File "$OutDir\gate.log" -Append -Encoding utf8; $failed += "$tag (server not ready)"; continue }
      "$tag ready" | Out-File "$OutDir\gate.log" -Append -Encoding utf8
      python $client $Port "$OutDir\$tag.json" $root seq 2>&1 | Out-File "$OutDir\$tag.client.txt" -Encoding utf8
      $clientExit = $LASTEXITCODE
      "$tag client exit $clientExit" | Out-File "$OutDir\gate.log" -Append -Encoding utf8
      if ($clientExit -ne 0) { $failed += "$tag (client exit $clientExit)" }
    } finally {
      if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit(120000) | Out-Null }
    }
  }
  if ($failed.Count -gt 0) {
    "FAILED: $($failed -join '; ')" | Out-File "$OutDir\gate.log" -Append -Encoding utf8
    $exitCode = 1
  } else {
    "ALL DONE" | Out-File "$OutDir\gate.log" -Append -Encoding utf8
  }
} finally { $lock.Dispose() }
exit $exitCode
