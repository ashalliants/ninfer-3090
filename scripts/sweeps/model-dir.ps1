# Where the .ninfer artifacts are, for the sweeps in this directory. Dot-source it and call
# Get-NInferModelDir; every sweep here does.
#
# A checkout can hold artifacts in two places and both are legitimate:
#
#   * the repository's own models\ -- what .gitignore has always ignored, and where a large local
#     collection tends to end up because it sits beside build-ninja\ rather than inside scripts\;
#   * scripts\models\ -- where scripts\download-qwen*.{bat,sh} put them by default, and what
#     docs/rtx-3090-linux.md documents as the download location.
#
# A sweep that hardcodes either one fails for whoever followed the other instruction, so probe:
# the first candidate that actually contains an artifact wins, repository root first. Testing for
# *.ninfer rather than for the directory matters -- an empty models\ directory left behind by a
# cleared download would otherwise shadow a populated one.
#
# NINFER_MODEL_DIR is taken verbatim and never probed. Falling back past a directory the caller
# named turns their typo into a missing-artifact error about a path they never mentioned.
#
# Do not move this logic into a param() default in a calling script: $PSScriptRoot is empty while a
# param default is being bound, so a repo-relative path written there resolves against the drive
# root and silently yields C:\models. Measured, not assumed.
function Get-NInferModelDir {
    if ($env:NINFER_MODEL_DIR) { return $env:NINFER_MODEL_DIR }
    # $PSScriptRoot inside a function is the directory of the file that *defined* it, which is this
    # one, regardless of which sweep dot-sourced it or what the current location is.
    $candidates = @(
        [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\models')),
        [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\models'))
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -Path (Join-Path $candidate '*.ninfer')) { return $candidate }
    }
    # Neither holds an artifact. Return the repository's own models\ so the caller's own
    # "model not found" throw names the conventional location instead of an empty string.
    return $candidates[0]
}
