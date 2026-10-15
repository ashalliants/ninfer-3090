param([string[]]$Targets, [switch]$Configure)
$exitCode = 0
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))
if ($Configure) {
  cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON `
   -DCMAKE_TOOLCHAIN_FILE="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" `
   -DVCPKG_INSTALLED_DIR=C:/ninfer-fork/ninfer-3090/build-ninja/vcpkg_installed -DVCPKG_MANIFEST_INSTALL=OFF 2>&1 | Select-Object -Last 15
  if ($LASTEXITCODE -ne 0) { $exitCode = $LASTEXITCODE }
}
if ($exitCode -eq 0 -and $Targets) {
  cmake --build build-ninja --target @Targets 2>&1 | Select-Object -Last 60
  if ($LASTEXITCODE -ne 0) { $exitCode = $LASTEXITCODE }
}
exit $exitCode

