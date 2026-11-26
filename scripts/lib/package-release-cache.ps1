# Reads build options from a build tree's CMakeCache.txt for package-release.ps1. Dot-source this
# file; it defines a function and runs nothing, so tests\release\test-package-release-cache.ps1 can
# drive it on fake caches.

# Test-CacheOptionOn: whether the cache sets the option to a value the CMakeLists' `if(NAME)` treats
# as true: any value but CMake's false constants (empty, 0, OFF, NO, FALSE, N or IGNORE in any case,
# or NOTFOUND or *-NOTFOUND in uppercase only), so =TRUE or =1 is caught as well as =ON. No cache, or
# no such entry, counts as off. Mirrors cache_option_on in package-release.sh.
function Test-CacheOptionOn([string]$BuildRoot, [string]$Name) {
    $cache = Join-Path $BuildRoot 'CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $cache)) { return $false }
    $entry = Select-String -LiteralPath $cache -CaseSensitive -Pattern "^$([regex]::Escape($Name)):[A-Za-z]*=(.*)$" | Select-Object -First 1
    if (-not $entry) { return $false }
    $value = $entry.Matches[0].Groups[1].Value
    if ($value -cmatch '^(.*-)?NOTFOUND$') { return $false }
    return -not ($value -imatch '^(|0|OFF|NO|FALSE|N|IGNORE)$')
}
