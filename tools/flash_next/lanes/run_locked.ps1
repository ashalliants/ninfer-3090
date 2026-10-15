param([string]$Log, [string]$Exe, [string]$ArgsFile)
# Holds the shared GPU lock (an open handle with no sharing) for the whole run of $Exe.
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
$exitCode = 0
$waitStart = Get-Date
while (-not $lock) {
  try   { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
  catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }
}
try {
  "lock acquired after {0:N0} s" -f ((Get-Date) - $waitStart).TotalSeconds | Out-File $Log -Encoding utf8
  $os = Get-CimInstance Win32_OperatingSystem
  "free RAM GiB: {0:N1}" -f ($os.FreePhysicalMemory/1MB) | Out-File $Log -Append -Encoding utf8
  $env:NINFER_TEST_QWEN4EXP_ARTIFACT = 'C:\ninfer-fork\ninfer-3090\models\qwen4exp-iq2xs-mtp-vision.ninfer'
  $ArgList = @((Get-Content $ArgsFile -Raw) -split "\s+" | Where-Object { $_ }); & $Exe @ArgList 2>&1 | Out-File $Log -Append -Encoding utf8
  $exitCode = $LASTEXITCODE
  "exit code $exitCode" | Out-File $Log -Append -Encoding utf8
} finally {
  $lock.Dispose()
}
exit $exitCode



