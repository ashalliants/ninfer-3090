# One step of the perplexity-drift bisect: build whatever is checked out, score int8, print it.
#
# TODO section 3: `int8` scored 4.343263 when the figures were published in 838c8b5d (2026-08-29)
# and 4.342425 at master. Same corpus, same window, same 261,167 scored tokens, and the artifact
# has not been reconverted since before the old measurement -- so something between the two changed
# arithmetic in some kernel.
#
# THERE ARE TWO TRANSITIONS IN THAT RANGE, NOT ONE. 66378f06 scores 4.342864374753507, a third
# value: 838c8b5d -> 66378f06 is -0.0092% and 66378f06 -> HEAD is -0.0101%. Bisect the two halves
# separately; a single bisect will converge on one of them and name a commit that cannot reproduce
# the whole drift.
#
# START THE BISECT `--first-parent`. The naive walk descends into the upstream catch-up branch,
# where CMake refuses with "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got '86'" (e.g.
# 00f02055). Those commits are unbuildable on this card -- not flaky, unbuildable -- and
# --first-parent both avoids them and cuts the range from 477 commits to 96.
#
# COPY THIS FILE OUT OF THE REPO BEFORE YOU START. `git bisect` checks out commits that predate it,
# which deletes it mid-run. Run the copy, pass -Repo, and it will Set-Location there itself.
#
# Run from (or pointed at) a detached HEAD under `git bisect`, then classify:
#
#   4.343263...  -> old behaviour  -> git bisect good
#   4.342425...  -> new behaviour  -> git bisect bad
#   build failed -> git bisect skip
#
# Deliberately prints the full precision. The two values differ in the fifth decimal, and a
# four-figure print would make them look identical and the bisect would converge on nothing.
[CmdletBinding()]
param(
  [string]$Repo   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$Model  = 'models\qwen3_8_27b.ninfer',
  [string]$Out    = 'profiles\ppl_bisect',
  [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
  [string]$Cuda   = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe'
)
$ErrorActionPreference = 'Continue'
Set-Location $Repo

$sha = (git rev-parse --short HEAD).Trim()
"== bisect step at $sha  ($(git log -1 --format='%ad %s' --date=short))"

# Build ONLY ninfer-perplexity. The default target also builds the test binaries, and those break
# independently of the thing being measured -- at de5fc15a, tests/ops/softmax_attention/
# causal_cache.cpp fails under MSVC with "error C3493: 'order' cannot be implicitly captured".
# Building everything would have skipped a perfectly measurable commit. It is also much faster.
#
# Redirect the build whole, then grep -- never truncate a build pipeline. And always delete
# build.log first: a killed build leaves the PREVIOUS run's BUILD_EXIT=0 behind and the step reads
# a stale success.
Remove-Item build.log -ErrorAction SilentlyContinue
$bat = Join-Path $env:TEMP "ninfer_bisect_build_$PID.bat"
@(
  '@echo off'
  "call `"$VcVars`" >nul 2>&1"
  "set CUDACXX=$Cuda"
  'cmake --build build-ninja --target ninfer-perplexity > build.log 2>&1'
  'echo BUILD_EXIT=%ERRORLEVEL% >> build.log'
) | Set-Content -LiteralPath $bat -Encoding ASCII
try { & cmd.exe /c $bat | Out-Null } finally { Remove-Item $bat -ErrorAction SilentlyContinue }

$m = (Select-String -Path build.log -Pattern 'BUILD_EXIT=(\d+)' -ErrorAction SilentlyContinue)
if (-not $m -or $m.Matches.Groups[1].Value -ne '0') {
  "   BUILD FAILED at $sha -- classify with: git bisect skip"
  Select-String -Path build.log -Pattern 'error|FAILED' | Select-Object -First 3 | ForEach-Object { "     $($_.Line)" }
  exit 2
}

Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
nvidia-smi -lgc 1500 | Out-Null
try {
  # No --log-level: it postdates part of the bisect range and older builds reject it outright.
  & .\build-ninja\apps\ninfer-perplexity.exe $Model `
      --corpus 'eval\corpora\perplexity-1m\manifest.json' --quick `
      --context 4096 --stride 2048 --kv-dtype int8 `
      --output $Out > "$Out.log" 2>&1
  $code = $LASTEXITCODE
} finally { nvidia-smi -rgc | Out-Null }

$report = Get-ChildItem -Recurse -Filter report.json $Out -ErrorAction SilentlyContinue | Select-Object -First 1
if ($code -ne 0 -or -not $report) {
  "   RUN FAILED at $sha (exit $code) -- classify with: git bisect skip"
  Get-Content "$Out.log" -Tail 3 -ErrorAction SilentlyContinue | ForEach-Object { "     $_" }
  exit 3
}
$j = Get-Content $report.FullName -Raw | ConvertFrom-Json
"   $sha  ppl = $($j.overall.perplexity)  tokens = $($j.overall.scored_tokens)"
"   838c8b5d = 4.343263 | 66378f06 = 4.342864 | HEAD = 4.342425"
