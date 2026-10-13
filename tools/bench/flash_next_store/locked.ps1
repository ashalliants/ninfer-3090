# Runs a script block file under the shared GPU lock (CLAUDE.md protocol): acquire, run, release.
param([Parameter(Mandatory = $true)][string]$Script, [string]$Log)
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
$waitStart = Get-Date
while (-not $lock) {
  try   { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
  catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }
}
"lock acquired after $([int]((Get-Date) - $waitStart).TotalSeconds) s at $(Get-Date -Format o)"
try {
  $ram = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
  "free RAM $ram GiB; GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)"
  & $Script
} finally {
  $lock.Dispose()
  "lock released at $(Get-Date -Format o)"
}
