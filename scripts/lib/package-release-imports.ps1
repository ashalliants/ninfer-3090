# The static import check of package-release.ps1, used when the packaging machine has no NVIDIA
# driver and so cannot start the executables. Dot-source this file; it defines functions and runs
# nothing, so tests\release\test-package-release-imports.ps1 can drive them with dumpbin output
# captured in fixtures.
#
# Every DLL any .exe or .dll in the archive folder imports must be in that folder, be a Windows
# system DLL, be an API set, or be one of the NVIDIA driver's own DLLs. The archive is flat, so
# "in the folder" means a file directly in it: a DLL in a subfolder is not found by the loader.

$DriverDlls = @('nvcuda.dll', 'nvml.dll', 'nvapi64.dll', 'nvcuvid.dll', 'nvencodeapi64.dll')

function Find-Dumpbin {
    $onPath = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    foreach ($root in @("${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022", "$env:ProgramFiles\Microsoft Visual Studio\2022")) {
        $hit = Get-ChildItem -Path "$root\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    throw 'dumpbin.exe not found: it ships with the VS 2022 C++ build tools, which the release build needs anyway'
}

# `dumpbin /nologo /dependents` output for one image, as lines.
function Get-DumpbinReport([string]$ImagePath) {
    $report = & (Find-Dumpbin) /nologo /dependents $ImagePath
    if ($LASTEXITCODE -ne 0) { throw "dumpbin failed on $(Split-Path -Leaf $ImagePath) (exit $LASTEXITCODE)" }
    return $report
}

# The DLL names (lower case) listed in a dumpbin /dependents report. They are the lines indented
# by exactly four spaces and ending in .dll: the header, "Summary" and its section sizes use other
# indents, and a DLL named in a "Dump of file" line is not indented at all.
function Get-DumpbinDependents([string[]]$Report) {
    foreach ($line in $Report) {
        if ($line -match '^\s{4}(\S+\.dll)\s*$') { $Matches[1].ToLowerInvariant() }
    }
}

# API sets are resolved by the loader and have no file in System32.
function Test-ApiSetDll([string]$Name) {
    return $Name -like 'api-ms-win-*' -or $Name -like 'ext-ms-*'
}

function Test-SystemDllInstalled([string]$Name) {
    return Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32\$Name")
}

# The lower-case names of the files directly in the archive folder (the layout is flat).
function Get-ArchiveFileSet([string]$Root) {
    $shipped = @{}
    Get-ChildItem -LiteralPath $Root -File | ForEach-Object { $shipped[$_.Name.ToLowerInvariant()] = $true }
    return $shipped
}

# The imports in one report that nothing accounts for. IsSystemDll is injectable so a test does
# not need a Windows System32.
function Get-UnsatisfiedDependents([string[]]$Report, [hashtable]$Shipped, [scriptblock]$IsSystemDll = { param($n) Test-SystemDllInstalled $n }) {
    foreach ($dependency in @(Get-DumpbinDependents $Report)) {
        if ($Shipped.ContainsKey($dependency)) { continue }
        if ($DriverDlls -contains $dependency) { continue }
        if (Test-ApiSetDll $dependency) { continue }
        if (& $IsSystemDll $dependency) { continue }
        $dependency
    }
}

# Throws when any .exe or .dll in $Root imports something the archive does not account for.
# ReadDependents maps an image's path to its dumpbin report; the packager passes Get-DumpbinReport.
function Assert-ArchiveDependenciesSatisfied([string]$Root, [scriptblock]$ReadDependents, [scriptblock]$IsSystemDll = { param($n) Test-SystemDllInstalled $n }) {
    $shipped = Get-ArchiveFileSet $Root
    $problems = [System.Collections.Generic.List[string]]::new()
    $images = @(Get-ChildItem -LiteralPath $Root -File | Where-Object { $_.Extension -in '.exe', '.dll' })
    if ($images.Count -eq 0) { throw "No executables or DLLs found in $Root to check" }
    foreach ($image in $images) {
        $report = & $ReadDependents $image.FullName
        foreach ($dependency in @(Get-UnsatisfiedDependents $report $shipped $IsSystemDll)) {
            $problems.Add("$($image.Name) imports $dependency")
        }
    }
    if ($problems.Count -gt 0) {
        throw ("The archive is missing DLLs its binaries import (checked statically, no NVIDIA driver here):`n  " + ($problems -join "`n  "))
    }
    Write-Host "dependency check: every import of the $($images.Count) images is shipped, a Windows DLL or an NVIDIA driver DLL"
}
