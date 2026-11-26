# Tests Test-CacheOptionOn (scripts/lib/package-release-cache.ps1), the CMakeCache.txt reader behind
# package-release.ps1's refusal of a -lineinfo tree, against what CMake itself makes of each value
# (tests/release/cmake-truth-values.txt). Needs no build tree or Windows machine:
#
#   pwsh tests/release/test-package-release-cache.ps1
#
# Exits non-zero on the first failed assertion.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../../scripts/lib/package-release-cache.ps1')

$root = Join-Path ([System.IO.Path]::GetTempPath()) ("ninfer-cache-test-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $root | Out-Null
$cache = Join-Path $root 'CMakeCache.txt'
$checks = 0
function Assert-On([bool]$Expected, [string]$What) {
    $script:checks++
    $actual = Test-CacheOptionOn $root 'NINFER_CUDA_LINEINFO'
    if ($actual -ne $Expected) { throw "FAIL: $What`n  expected on=$Expected, got on=$actual" }
}
try {
    foreach ($line in Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cmake-truth-values.txt')) {
        if ($line.StartsWith('#')) { continue }
        $want, $value = $line -split "`t", 2
        Set-Content -LiteralPath $cache -Value "NINFER_CUDA_LINEINFO:BOOL=$value"
        Assert-On ($want -eq 'on') "value '$value' (CMake says $want)"
    }
    # Any cache type, and the entry among others.
    Set-Content -LiteralPath $cache -Value @('FOO:BOOL=OFF', 'NINFER_CUDA_LINEINFO:STRING=TRUE', 'BAR:BOOL=ON')
    Assert-On $true 'a STRING entry among others'
    # Another option whose name starts with this one's does not count.
    Set-Content -LiteralPath $cache -Value 'NINFER_CUDA_LINEINFO_X:BOOL=ON'
    Assert-On $false 'a longer name with the same prefix'
    # Names are case-sensitive in the cache.
    Set-Content -LiteralPath $cache -Value 'ninfer_cuda_lineinfo:BOOL=ON'
    Assert-On $false 'the name in another case'
    Remove-Item -LiteralPath $cache
    Assert-On $false 'no CMakeCache.txt'
} finally {
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Output "package-release cache reader: $checks checks passed"
