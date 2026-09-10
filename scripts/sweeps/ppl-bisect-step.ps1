# One step of the perplexity-drift bisect: build whatever is checked out, score int8, print it.
#
# TODO section 3: `int8` scored 4.343263 when the figures were published in 838c8b5d (2026-08-29)
# and 4.342425 at master. Same corpus, same window, same 261,167 scored tokens, and the artifact
# has not been reconverted since before the old measurement -- so something between the two changed
# arithmetic in some kernel. The range is ~477 commits, which is nine bisect steps.
#
# Run this from a detached HEAD under `git bisect`, then classify:
#
#   4.343263...  -> old behaviour  -> git bisect good
#   4.342425...  -> new behaviour  -> git bisect bad
#   build failed -> git bisect skip
#
# Deliberately prints the full precision. The two values differ in the fifth decimal, and a
# four-figure print would make them look identical and the bisect would converge on nothing.
[CmdletBinding()]
param(
  [string]$Model  = 'models\qwen3_8_27b.ninfer',
  [string]$Out    = 'profiles\ppl_bisect'
)
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

$sha = (git rev-parse --short HEAD).Trim()
"== bisect step at $sha"

# A full rebuild, because a bisect jumps hundreds of commits and ninja's incremental state is
# meaningless across that. Redirect whole, then grep -- never truncate a build pipeline.
Remove-Item build.log -ErrorAction SilentlyContinue
& cmd.exe /c "C:\ninfer-fork\ninfer-3090\build_only.bat" | Out-Null
$exit = (Select-String -Path build.log -Pattern 'BUILD_EXIT=(\d+)').Matches.Groups[1].Value
if ($exit -ne '0') {
  "   BUILD FAILED at $sha -- classify with: git bisect skip"
  Select-String -Path build.log -Pattern 'error|FAILED' | Select-Object -First 3 | ForEach-Object { "     $($_.Line)" }
  exit 2
}

Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
nvidia-smi -lgc 1500 | Out-Null
try {
  & .\build-ninja\apps\ninfer-perplexity.exe $Model `
      --corpus 'eval\corpora\perplexity-1m\manifest.json' --quick `
      --context 4096 --stride 2048 --kv-dtype int8 `
      --output $Out --log-level warning > "$Out.log" 2>&1
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
"   old = 4.343263 (bisect good) | new = 4.342425 (bisect bad)"
