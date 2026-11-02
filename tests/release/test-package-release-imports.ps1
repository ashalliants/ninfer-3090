# Tests the no-driver import check of scripts/package-release.ps1 on dumpbin output captured in
# tests/release/dumpbin, so it needs neither Windows, dumpbin.exe nor an NVIDIA driver:
#
#   pwsh tests/release/test-package-release-imports.ps1
#
# Exits non-zero on the first failed assertion.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../../scripts/lib/package-release-imports.ps1')

$Fixtures = Join-Path $PSScriptRoot 'dumpbin'
function Read-Fixture([string]$Name) { Get-Content -LiteralPath (Join-Path $Fixtures "$Name.txt") }

$script:Checks = 0
function Assert-Equal($Actual, $Expected, [string]$What) {
    $script:Checks++
    $a = (@($Actual) -join ',')
    $e = (@($Expected) -join ',')
    if ($a -ne $e) { throw "FAIL: $What`n  expected: [$e]`n  actual:   [$a]" }
}
function Assert-Throws([scriptblock]$Action, [string]$Pattern, [string]$What) {
    $script:Checks++
    try { & $Action } catch {
        if ($_.Exception.Message -notmatch $Pattern) { throw "FAIL: $What`n  message [$($_.Exception.Message)] does not match /$Pattern/" }
        return
    }
    throw "FAIL: $What`n  nothing was thrown"
}

# A System32 that holds exactly these DLLs.
function New-SystemDirectory([string[]]$Names) {
    $set = @{}
    $Names | ForEach-Object { $set[$_.ToLowerInvariant()] = $true }
    return { param($n) $set.ContainsKey($n) }.GetNewClosure()
}
$NoSystemDlls = New-SystemDirectory @()
$Windows      = New-SystemDirectory @('kernel32.dll', 'user32.dll', 'bcrypt.dll')

# --- the parser ---------------------------------------------------------------------------------
# Names are lower-cased; the "Dump of file" path, the section sizes in Summary (one is named
# .nvFatBi.dll) and the "Image has ..." headers are not imports; delay-load imports are.
Assert-Equal (Get-DumpbinDependents (Read-Fixture 'shipped')) @('cublas64_12.dll', 'avcodec-62.dll') 'shipped: names, lower-cased, no summary or header lines'
Assert-Equal (Get-DumpbinDependents (Read-Fixture 'driver')) @('nvcuda.dll', 'nvml.dll', 'nvapi64.dll', 'nvcuvid.dll', 'nvencodeapi64.dll') 'driver: includes delay-load imports'
Assert-Equal (Get-DumpbinDependents @()) @() 'an empty report has no imports'

# --- one report against the rules ---------------------------------------------------------------
$flat = @{ 'cublas64_12.dll' = $true; 'avcodec-62.dll' = $true; 'ninfer-serve.exe' = $true }
Assert-Equal (Get-UnsatisfiedDependents (Read-Fixture 'shipped') $flat $NoSystemDlls) @() 'shipped DLLs satisfy the check, whatever their case'
Assert-Equal (Get-UnsatisfiedDependents (Read-Fixture 'shipped') @{} $NoSystemDlls) @('cublas64_12.dll', 'avcodec-62.dll') 'the same imports fail when the archive does not carry them'
Assert-Equal (Get-UnsatisfiedDependents (Read-Fixture 'system') @{} $Windows) @() 'Windows system DLLs satisfy the check'
Assert-Equal (Get-UnsatisfiedDependents (Read-Fixture 'system') @{} $NoSystemDlls) @('kernel32.dll', 'bcrypt.dll', 'user32.dll') 'system DLLs are flagged on a machine without them'
Assert-Equal (Get-UnsatisfiedDependents (Read-Fixture 'apiset') @{} $NoSystemDlls) @() 'API sets need no file anywhere'
Assert-Equal (Get-UnsatisfiedDependents (Read-Fixture 'driver') @{} $NoSystemDlls) @() 'NVIDIA driver DLLs are not required in the archive'
Assert-Equal (Get-UnsatisfiedDependents (Read-Fixture 'missing') @{} $Windows) @('cublaslt64_12.dll', 'cudart64_12.dll') 'missing DLLs are reported; system and driver imports in the same report are not'

# --- the whole check on a flat archive folder ---------------------------------------------------
$images = @{
    'ninfer-serve.exe' = 'shipped'    # cublas64_12.dll, avcodec-62.dll
    'avcodec-62.dll'   = 'system'
    'cublas64_12.dll'  = 'driver'
}
$reader = { param($path) Read-Fixture $images[(Split-Path -Leaf $path)] }.GetNewClosure()

$root = Join-Path ([IO.Path]::GetTempPath()) "ninfer-imports-test-$([guid]::NewGuid())"
New-Item -ItemType Directory -Path $root | Out-Null
try {
    foreach ($name in $images.Keys) { New-Item -ItemType File -Path (Join-Path $root $name) | Out-Null }
    New-Item -ItemType File -Path (Join-Path $root 'README.md') | Out-Null   # not an image, not checked

    Assert-Equal (Get-ArchiveFileSet $root).Keys.Count 4 'the archive set holds the files of the folder'
    Assert-Equal ($null -ne (Get-ArchiveFileSet $root)['cublas64_12.dll']) $true 'the archive set is keyed by lower-case name'
    Assert-Equal ((Assert-ArchiveDependenciesSatisfied $root $reader $Windows 6>&1) -join '') 'dependency check: every import of the 3 images is shipped, a Windows DLL or an NVIDIA driver DLL' 'a complete archive passes'

    # A DLL in a subfolder is not on the loader's path for the executables beside it.
    $images['ninfer-bench.exe'] = 'missing'
    New-Item -ItemType File -Path (Join-Path $root 'ninfer-bench.exe') | Out-Null
    $reader = { param($path) Read-Fixture $images[(Split-Path -Leaf $path)] }.GetNewClosure()
    New-Item -ItemType Directory -Path (Join-Path $root 'lib') | Out-Null
    New-Item -ItemType File -Path (Join-Path (Join-Path $root 'lib') 'cublasLt64_12.dll') | Out-Null
    Assert-Throws { Assert-ArchiveDependenciesSatisfied $root $reader $Windows 6>$null } 'ninfer-bench\.exe imports cublaslt64_12\.dll\s+ninfer-bench\.exe imports cudart64_12\.dll$' 'a DLL in a subfolder does not satisfy an import, and each missing DLL is named with its image'

    # Shipping it flat fixes that one, and only that one.
    New-Item -ItemType File -Path (Join-Path $root 'cublasLt64_12.dll') | Out-Null
    $images['cublasLt64_12.dll'] = 'driver'
    $reader = { param($path) Read-Fixture $images[(Split-Path -Leaf $path)] }.GetNewClosure()
    Assert-Throws { Assert-ArchiveDependenciesSatisfied $root $reader $Windows 6>$null } 'ninfer-bench\.exe imports cudart64_12\.dll$' 'only the DLL still missing is reported'

    # A failure to read an image (dumpbin exiting non-zero) must stop the release, not be skipped.
    Assert-Throws { Assert-ArchiveDependenciesSatisfied $root { param($p) throw 'dumpbin failed on x (exit 1)' } $Windows 6>$null } 'dumpbin failed' 'a reader failure propagates'
} finally {
    Remove-Item -LiteralPath $root -Recurse -Force
}

$empty = Join-Path ([IO.Path]::GetTempPath()) "ninfer-imports-empty-$([guid]::NewGuid())"
New-Item -ItemType Directory -Path $empty | Out-Null
try {
    Assert-Throws { Assert-ArchiveDependenciesSatisfied $empty $reader $Windows } 'No executables or DLLs found' 'an archive with no images is an error, not a pass'
} finally {
    Remove-Item -LiteralPath $empty -Recurse -Force
}

Write-Host "ok: $script:Checks checks"
