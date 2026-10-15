# Builds targets in this checkout's existing build-ninja tree. It never configures or reconfigures:
# AGENTS.md says to use the already configured tree as it is. A checkout without one (a fresh
# worktree) needs the one-time configure recipe in AGENTS.md ("A worktree session needs its own
# build tree"), run by hand from the VS 2022 environment.
#
#   pwsh tools\flash_next\lanes\build.ps1 -Targets ninfer_tests, ninfer-serve
param([Parameter(Mandatory = $true)][string[]]$Targets)
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$tree = Join-Path $root 'build-ninja'
if (-not (Test-Path (Join-Path $tree 'CMakeCache.txt'))) {
  Write-Error "No configured build tree at $tree. Configure it once with the recipe in AGENTS.md (A worktree session needs its own build tree); this script does not configure."
  exit 2
}
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
Set-Location $root
cmake --build build-ninja --target @Targets 2>&1 | Select-Object -Last 60
exit $LASTEXITCODE
