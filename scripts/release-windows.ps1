# Build and package the Windows release on a Windows machine that has NO NVIDIA GPU.
#
#   git clone https://github.com/ashalliants/ninfer-3090.git C:\ninfer-release\src
#   cd C:\ninfer-release\src ; git checkout --detach <release commit>
#   powershell -ExecutionPolicy Bypass -File scripts\release-windows.ps1
#
# Compiling CUDA needs the toolkit, not a card. This installs whatever is missing (VS 2022 Build
# Tools, CMake, Ninja, CUDA 12.8.1, vcpkg at the manifest's pinned baseline, ccache), builds
# ninfer, ninfer-serve and ninfer_bench from the checked-out commit, and runs the packager.
# Safe to run again: every step checks first, and the build tree, vcpkg binary cache and ccache
# make a second run incremental. Needs an elevated shell for the installs (-SkipBootstrap skips
# them once the machine is set up).
#
# CUDA 12.8.1 is deliberate: it is what the Linux release is built with (the vastai/base-image
# cuda-12.8.1 image), and the Windows archive ships cublas64_12.dll from this toolkit, so keeping
# the two platforms on one toolkit means one set of cuBLAS behaviour to reason about.
#
# What this cannot do: there is no driver here, so the executables are never run. The packager
# falls back to checking import tables (see package-release.ps1) and says so. Smoke-test the
# published zip on the RTX 3090 when you can, and say in the release notes that it was not run.
#
# Progress goes to the console and to <WorkDir>\release-windows.log. The last line of
# <WorkDir>\STATUS.txt is DONE or FAILED, which is what a remote driver polls.
[CmdletBinding()]
param(
    [string]$WorkDir = 'C:\ninfer-release',
    [switch]$SkipBootstrap
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$RepoRoot = Split-Path -Parent $PSScriptRoot

$CudaVersion = '12.8.1'
$CudaDriverVersion = '572.61'
$CudaInstallerBytes = 3369253408
# SHA-256 of the installer at the URL below, computed from the copy used for the v0.16.1 build. NVIDIA's page
# lists no SHA-256, so this pins what was fetched over TLS; it is checked before the installer is run elevated.
$CudaInstallerSha256 = '19392bbffd0ad4ee7cb295a181e87f682187f17653679c1c548c263b7e1cd9a6'
$CudaInstallerUrl = "https://developer.download.nvidia.com/compute/cuda/$CudaVersion/local_installers/cuda_${CudaVersion}_${CudaDriverVersion}_windows.exe"
$CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
# Compiler and library components only. The display driver is the one thing left out on purpose.
$CudaComponents = @('nvcc_12.8', 'cudart_12.8', 'cublas_12.8', 'cublas_dev_12.8', 'cccl_12.8',
                    'nvrtc_12.8', 'nvrtc_dev_12.8', 'nvtx_12.8', 'cuda_profiler_api_12.8', 'nvml_dev_12.8')

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$StatusFile = Join-Path $WorkDir 'STATUS.txt'
Set-Content -LiteralPath $StatusFile -Value 'RUNNING'
Start-Transcript -Path (Join-Path $WorkDir 'release-windows.log') -Append | Out-Null

function Write-Step([string]$Message) { Write-Host ''; Write-Host "==> $(Get-Date -Format s) $Message" }

function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

function Assert-Admin {
    $principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this from an elevated (Administrator) PowerShell, or pass -SkipBootstrap if everything is already installed.'
    }
}

function Install-WithWinget([string]$Id, [string]$Override) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is missing, so $Id cannot be installed automatically. Install 'App Installer' from the Microsoft Store, or install $Id by hand."
    }
    $wingetArgs = @('install', '--id', $Id, '-e', '--silent', '--accept-package-agreements', '--accept-source-agreements')
    if ($Override) { $wingetArgs += @('--override', $Override) }
    & winget @wingetArgs
    # 3010 is "installed, reboot pending". Carry on: nothing below needs the reboot unless it fails.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) { throw "winget install $Id failed ($LASTEXITCODE)" }
    Update-SessionPath
}

function Find-VcVars {
    @('C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
      'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat') |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

function Test-CudaToolkit {
    $required = @('bin\nvcc.exe', 'bin\cublas64_12.dll', 'bin\cublasLt64_12.dll', 'lib\x64\cuda.lib', 'lib\x64\cudart_static.lib',
                  'include\nvtx3', 'EULA.txt')
    $missing = $required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $CudaRoot $_)) }
    return @($missing)
}

# The NVIDIA installer is a 7z self-extractor. On a machine with no NVIDIA GPU its silent mode has
# been seen to exit after about two seconds (right after "No driver installed for the display
# device") and install nothing. The toolkit components are plain directories in the archive, so
# copying them is the same result without the installer. 7zr.exe is a 600 KB standalone download.
function Expand-CudaToolkit([string]$Installer, [string]$Destination) {
    # Pinned release and digest: this runs elevated, so a cached or downloaded copy is checked every time.
    $sevenZipSha256 = '9bf582b894dfa665f852d7e9112b02290a1c303c2d95a21fcdc41ffcb03143b7'
    $sevenZip = Join-Path $WorkDir '7zr.exe'
    if (-not (Test-Path -LiteralPath $sevenZip) -or (Get-FileHash -LiteralPath $sevenZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sevenZipSha256) {
        & curl.exe -L --fail --retry 5 -o $sevenZip 'https://github.com/ip7z/7zip/releases/download/26.05/7zr.exe'
        if ($LASTEXITCODE -ne 0) { throw "7zr download failed ($LASTEXITCODE)" }
    }
    $actual = (Get-FileHash -LiteralPath $sevenZip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $sevenZipSha256) {
        Remove-Item -LiteralPath $sevenZip -Force
        throw "7zr.exe checksum mismatch (expected $sevenZipSha256, got $actual)"
    }
    $stage = Join-Path $WorkDir 'cuda-extract'
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
    # Package directory in the archive -> merged into the toolkit root. EULA.txt sits at the archive root.
    $packages = @('cuda_nvcc\nvcc', 'cuda_cudart\cudart', 'libcublas\cublas', 'libcublas\cublas_dev', 'cuda_cccl\thrust',
                  'cuda_nvrtc\nvrtc', 'cuda_nvrtc\nvrtc_dev', 'cuda_nvtx\nvtx', 'cuda_profiler_api\cuda_profiler_api',
                  'cuda_nvml_dev\nvml_dev', 'cuda_cuobjdump\cuobjdump', 'cuda_nvdisasm\nvdisasm', 'cuda_cuxxfilt\cuxxfilt',
                  'cuda_nvprune\nvprune')
    $include = @('EULA.txt') + ($packages | ForEach-Object { "$_\*" })
    & $sevenZip x -y "-o$stage" $Installer @include | Out-Null
    if ($LASTEXITCODE -gt 1) { throw "7zr extraction failed ($LASTEXITCODE)" }   # 1 is a warning (the SFX stub)
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -LiteralPath (Join-Path $stage 'EULA.txt') -Destination $Destination -Force
    foreach ($package in $packages) {
        Copy-Item -Path (Join-Path $stage "$package\*") -Destination $Destination -Recurse -Force
    }
    Remove-Item -LiteralPath $stage -Recurse -Force
}

function Install-Cuda {
    $installer = Join-Path $WorkDir "cuda_${CudaVersion}_windows.exe"
    if (-not (Test-Path -LiteralPath $installer) -or (Get-Item -LiteralPath $installer).Length -ne $CudaInstallerBytes) {
        Write-Step "downloading CUDA $CudaVersion (3.4 GB)"
        & curl.exe -L --fail --retry 5 --retry-delay 10 -C - -o $installer $CudaInstallerUrl
        if ($LASTEXITCODE -ne 0) { throw "CUDA download failed ($LASTEXITCODE)" }
        if ((Get-Item -LiteralPath $installer).Length -ne $CudaInstallerBytes) {
            throw "CUDA installer is the wrong size: expected $CudaInstallerBytes bytes. Delete $installer and re-run."
        }
    }
    $actualSha256 = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $CudaInstallerSha256) {
        throw "CUDA installer checksum mismatch (expected $CudaInstallerSha256, got $actualSha256). Delete $installer and re-run; do not run it."
    }
    Write-Step 'installing CUDA components (no driver)'
    $proc = Start-Process -FilePath $installer -ArgumentList (@('-s') + $CudaComponents) -PassThru
    if (-not $proc.WaitForExit(40 * 60 * 1000)) {
        Stop-Process -Id $proc.Id -Force
        throw 'The CUDA installer did not finish in 40 minutes (a silent install with an explicit component list has been seen to hang). Run the installer by hand once, choosing Custom and unticking the driver, then re-run with -SkipBootstrap.'
    }
    # Installer exit codes are unreliable; the file check below is the real test.
    Write-Host "CUDA installer exited $($proc.ExitCode)"
    if ((Test-CudaToolkit).Count -gt 0) {
        Write-Host 'The installer left the toolkit incomplete; unpacking the components from the installer instead.'
        Expand-CudaToolkit $installer $CudaRoot
    }
}

try {
    Write-Step "release commit $(git -C $RepoRoot rev-parse HEAD)"
    if (git -C $RepoRoot status --porcelain) {
        throw 'The checkout has uncommitted or untracked files (ignored files such as dist are fine). Release builds come from a clean commit.'
    }
    $ReleaseTag = (Get-Content -LiteralPath (Join-Path $RepoRoot 'VERSION') -Raw).Trim()
    Write-Host "VERSION: $ReleaseTag"

    $freeGb = [math]::Round((Get-PSDrive -Name ($WorkDir.Substring(0, 1))).Free / 1GB)
    Write-Host "free disk on $($WorkDir.Substring(0, 2)): $freeGb GB"
    if ($freeGb -lt 50) { throw 'Under 50 GB free: the CUDA toolkit, vcpkg ffmpeg build and the build tree need about that.' }

    if (-not $SkipBootstrap) {
        Write-Step 'bootstrapping the toolchain'
        Assert-Admin

        if (-not (Find-VcVars)) {
            Write-Step 'installing VS 2022 Build Tools (C++ workload)'
            Install-WithWinget 'Microsoft.VisualStudio.2022.BuildTools' `
                '--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621'
        }
        if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Install-WithWinget 'Kitware.CMake' }
        if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { Install-WithWinget 'Ninja-build.Ninja' }

        if ((Test-CudaToolkit).Count -gt 0) { Install-Cuda }
    }

    $missing = Test-CudaToolkit
    if ($missing.Count -gt 0) { throw "CUDA 12.8 toolkit at $CudaRoot is incomplete; missing: $($missing -join ', ')" }
    $env:CUDA_PATH = $CudaRoot
    $env:CUDA_PATH_V12_8 = $CudaRoot
    if (-not (Find-VcVars)) { throw 'VS 2022 Build Tools are not installed (see AGENTS.md, Windows build environment).' }
    foreach ($tool in 'cmake', 'ninja', 'git') {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not on PATH (open a new shell after installing it)" }
    }
    Write-Host (& "$CudaRoot\bin\nvcc.exe" --version | Select-String 'release')

    Write-Step 'vcpkg at the manifest baseline'
    $vcpkgRoot = Join-Path $WorkDir 'vcpkg'
    $baseline = (Get-Content -LiteralPath (Join-Path $RepoRoot 'vcpkg.json') -Raw | ConvertFrom-Json).'builtin-baseline'
    if (-not (Test-Path -LiteralPath (Join-Path $vcpkgRoot '.git'))) {
        git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
        if ($LASTEXITCODE -ne 0) { throw 'vcpkg clone failed' }
    }
    git -C $vcpkgRoot fetch --quiet origin
    git -C $vcpkgRoot checkout --quiet $baseline
    if ($LASTEXITCODE -ne 0) { throw "vcpkg checkout of $baseline failed" }
    if (-not (Test-Path -LiteralPath (Join-Path $vcpkgRoot 'vcpkg.exe'))) {
        & (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
        if ($LASTEXITCODE -ne 0) { throw 'vcpkg bootstrap failed' }
    }
    $env:VCPKG_ROOT = $vcpkgRoot
    $env:VCPKG_DISABLE_METRICS = '1'

    Write-Step 'ccache'
    & (Join-Path $PSScriptRoot 'setup-ccache.ps1')
    Update-SessionPath

    Write-Step 'configure, build and package (build.ps1 -Package)'
    # The first run builds ffmpeg from source through vcpkg: expect tens of minutes of quiet.
    & (Join-Path $PSScriptRoot 'build.ps1') -Package -BuildDir (Join-Path $WorkDir 'build')

    Write-Step 'result'
    $dist = Join-Path $RepoRoot 'dist'
    Get-ChildItem -LiteralPath $dist -File | Where-Object { $_.Name -like '*windows*' } |
        ForEach-Object { '{0}  {1:N0} MB  sha256 {2}' -f $_.Name, ($_.Length / 1MB), (Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
    Write-Host ''
    Write-Host 'NOT VERIFIED HERE: no NVIDIA driver, so the executables were not started and nothing was served.'
    Write-Host 'Record that in the release notes, and smoke-test the published zip on the RTX 3090.'
    Add-Content -LiteralPath $StatusFile -Value 'DONE'
}
catch {
    Write-Host ''
    Write-Host "FAILED: $($_.Exception.Message)"
    Add-Content -LiteralPath $StatusFile -Value 'FAILED'
    Stop-Transcript | Out-Null
    exit 1
}
Stop-Transcript | Out-Null
