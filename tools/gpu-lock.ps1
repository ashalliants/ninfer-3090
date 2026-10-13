<#
The shared GPU lock, as one reusable helper (protocol and reasons: CLAUDE.md, "GPU lock").

  . "$PSScriptRoot\gpu-lock.ps1"            # from another script
  Invoke-WithGpuLock { ...GPU work... }     # wait, run the block, always release
  $lock = Enter-GpuLock; try { ... } finally { Exit-GpuLock $lock }
  Test-GpuLockFree                          # a peek; the answer can change a moment later

The lock is an open file handle with no sharing, not the file's existence. The file stays on disk
for good: open it with OpenOrCreate and FileShare.None, never create it fresh and never delete it.
Windows closes the handle when its process exits, so a killed holder cannot leave it stuck.
Acquire and release inside ONE script run: a handle opened in one tool call is gone by the next.
Hold it for as short a time as the work allows. A suite of unit tests is many short jobs, not one
long one, so take it per test (tools/run-tests.ps1 does) and let other waiters in between.
#>

$script:GpuLockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'

function Enter-GpuLock {
  param(
    [int] $PollMilliseconds = 500,   # the lock has no queue; a slow poller loses to a fast one
    [scriptblock] $OnWait            # called once, when the first attempt finds it held
  )
  $told = $false
  while ($true) {
    try {
      return [IO.File]::Open($script:GpuLockPath, 'OpenOrCreate', 'ReadWrite', 'None')
    } catch [System.IO.IOException] {
      if (-not $told -and $OnWait) { & $OnWait; $told = $true }
      Start-Sleep -Milliseconds $PollMilliseconds
    }
  }
}

function Exit-GpuLock {
  param($Lock)
  if ($Lock) { $Lock.Dispose() }
}

function Invoke-WithGpuLock {
  param([Parameter(Mandatory)] [scriptblock] $Script, [scriptblock] $OnWait)
  $lock = Enter-GpuLock -OnWait $OnWait
  try { & $Script } finally { Exit-GpuLock $lock }
}

function Test-GpuLockFree {
  try {
    $probe = [IO.File]::Open($script:GpuLockPath, 'OpenOrCreate', 'ReadWrite', 'None')
    $probe.Dispose()
    return $true
  } catch [System.IO.IOException] { return $false }
}
