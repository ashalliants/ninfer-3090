param([string[]]$Targets, [switch]$Configure)
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
Set-Location C:\ninfer-fork\ninfer-3090\.claude\worktrees\linked-baking-piglet
if ($Configure) {
  cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON `
   -DVCPKG_INSTALLED_DIR=C:/ninfer-fork/ninfer-3090/build-ninja/vcpkg_installed -DVCPKG_MANIFEST_INSTALL=OFF 2>&1 | Select-Object -Last 15
}
if ($Targets) { cmake --build build-ninja --target @Targets 2>&1 | Select-Object -Last 60 }
