$ErrorActionPreference = 'Stop'

$ReleaseTag = '0.9.1-rtx3090'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = if ($env:NINFER_BUILD_ROOT) { $env:NINFER_BUILD_ROOT } else { Join-Path $RepoRoot 'build-ninja' }
$DistRoot = Join-Path $RepoRoot 'dist'
$ProductName = "ninfer-rtx3090-windows-x64-$ReleaseTag"
$ProductRoot = Join-Path $DistRoot $ProductName
$ArchivePath = Join-Path $DistRoot "$ProductName.zip"
$ChecksumPath = Join-Path $DistRoot 'SHA256SUMS-v0.9.1-windows.txt'

# Ninja is a single-config generator, so release binaries land directly under
# apps\ / bench\ rather than an apps\Release\ subdirectory.
$Products = @(
    @{ Source = 'apps\ninfer.exe'; Destination = 'ninfer.exe' },
    @{ Source = 'apps\ninfer-serve.exe'; Destination = 'ninfer-serve.exe' },
    @{ Source = 'bench\ninfer_bench.exe'; Destination = 'ninfer_bench.exe' }
)

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
$resolvedDist = (Resolve-Path -LiteralPath $DistRoot).Path
$resolvedProductParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $ProductRoot))
if ($resolvedProductParent -ne $resolvedDist -or (Split-Path -Leaf $ProductRoot) -ne $ProductName) {
    throw "Refusing to package outside the expected dist directory: $ProductRoot"
}
if (Test-Path -LiteralPath $ProductRoot) { Remove-Item -LiteralPath $ProductRoot -Recurse -Force }
if (Test-Path -LiteralPath $ArchivePath) { Remove-Item -LiteralPath $ArchivePath -Force }
New-Item -ItemType Directory -Path $ProductRoot | Out-Null

foreach ($product in $Products) {
    $source = Join-Path $BuildRoot $product.Source
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing release product: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $ProductRoot $product.Destination)
}
Get-ChildItem -LiteralPath (Join-Path $BuildRoot 'apps') -Filter '*.dll' | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $ProductRoot
}
Copy-Item -LiteralPath (Join-Path $RepoRoot 'VERSION') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $ProductRoot
# The archive README must describe the archive. docs\rtx-3090-windows.md is written for a checkout
# -- it points at scripts\download-qwen*.bat, and the packager copies those to the archive root --
# so a user following it from inside the archive got a missing-file error.
Copy-Item -LiteralPath (Join-Path $RepoRoot 'docs\release-archive-windows.md') -Destination (Join-Path $ProductRoot 'README.md')
Copy-Item -LiteralPath (Join-Path $RepoRoot 'RELEASE_NOTES_0.9.1.md') -Destination $ProductRoot
Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'scripts') -Filter '*.bat' | Where-Object Name -match '^(download|run)-qwen' |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $ProductRoot }

$innerHashes = Get-ChildItem -LiteralPath $ProductRoot -File | Sort-Object Name | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($_.Name)"
}
# LF, not CRLF. `Set-Content` writes CRLF on Windows, and `sha256sum -c` then looks for a file whose
# name ends in a carriage return: "avcodec-62.dll: FAILED open or read" for every line. The Linux
# archive's list is written by sha256sum itself and has always been LF, so this also makes the two
# archives' checksum files interchangeable. Get-FileHash comparisons are unaffected either way.
[IO.File]::WriteAllText((Join-Path $ProductRoot 'SHA256SUMS.txt'), (($innerHashes -join "`n") + "`n"), [Text.ASCIIEncoding]::new())
Compress-Archive -LiteralPath $ProductRoot -DestinationPath $ArchivePath -CompressionLevel Optimal
$archiveHash = Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256
[IO.File]::WriteAllText($ChecksumPath,
    "$($archiveHash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $ArchivePath)`n",
    [Text.ASCIIEncoding]::new())

Get-Item -LiteralPath $ArchivePath, $ChecksumPath |
    Select-Object Name, @{ Name = 'SizeMB'; Expression = { [math]::Round($_.Length / 1MB, 2) } }
