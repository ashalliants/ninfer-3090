# Does any DFlash2 draft count make it positive on *text*?
#
# TODO.md section 2c records DFlash2 costing ~20% against no speculation on the 27B, always measured
# with --draft-tokens 7, and asks whether a lower count crosses over. The obvious way to answer that
# -- sweep draft counts through ninfer_bench -- does not work, and finding out why is the point of
# this script existing separately.
#
# `bench/fixtures/bench_corpus.ids` holds 65,536 tokens drawn from **682 distinct ids**, with 98.4%
# of its bigrams repeated, because it is a curated bank tiled to length. Its own manifest says
# "repetition fills length only and does not bias throughput", which is true for plain decode and
# false for anything that drafts: a draft model predicts that text perfectly. Swept through
# ninfer_bench, DFlash2 reports **100% acceptance at every draft count from 1 to 12**, and decode
# rises monotonically to 159 tok/s because each round emits k+1 free tokens. None of that is a
# statement about text.
#
# So measure the serving path on the model's own generated continuation instead. The model writes
# natural prose in response to a real prompt, which is the distribution DFlash2 meets in production,
# and acceptance then reflects genuine drafting difficulty rather than a tiled fixture.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\dflash2-draft-tokens-realtext.ps1
#
# About 25 minutes. Greedy, so each configuration generates the same text and the comparison is of
# speed on identical output rather than of two different continuations.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

$modelDir = if ($env:NINFER_MODEL_DIR) { $env:NINFER_MODEL_DIR } else { 'C:\Ninefer-3090\models' }
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$weights = "$modelDir\qwen3_8_27b_dflash2.ninfer"
if (-not (Test-Path $weights)) { throw "Missing DFlash2 artifact: $weights" }

# A prompt that provokes a long, substantive, non-repetitive answer. Deliberately not a list or a
# table: those are exactly the shapes a draft head finds easy, and the question here is what
# ordinary prose costs.
$prompt = 'Explain how a paged key-value cache lets a transformer serve many concurrent requests ' +
          'without reserving each one its maximum context up front. Cover fragmentation, the ' +
          'block table indirection, and what happens when a request outgrows its allocation.'

$configs = @( @{ label='none'; args=@() } )
foreach ($n in 1,2,3,4,5,6,7,8,10,12) {
  $configs += @{ label="dflash2-$n";      args=@('--spec','dflash2','--draft-tokens',[string]$n) }
  $configs += @{ label="dflash2-$n+head"; args=@('--spec','dflash2','--draft-tokens',[string]$n,'--lm-head-draft') }
}
$configs += @{ label='mtp3';      args=@('--spec','mtp','--draft-tokens','3') }
$configs += @{ label='mtp3+head'; args=@('--spec','mtp','--draft-tokens','3','--lm-head-draft') }

"config,rep,decode_tok_s,generated_tokens"
foreach ($c in $configs) {
  for ($rep = 1; $rep -le 3; $rep++) {
    $log = "$out\rt_$($c.label -replace '\+','p')_$rep.log"
    $argv = @($weights,'--prompt',$prompt,'--max-new','256','--max-context','8192',
              '--kv-dtype','int8','--greedy','--no-thinking') + $c.args
    & .\build-ninja\apps\ninfer.exe @argv > $log 2>&1
    if ($LASTEXITCODE -ne 0) { "$($c.label),$rep,FAILED,"; Get-Content $log -Tail 2 | ForEach-Object { "    $_" }; continue }
    $txt = Get-Content $log -Raw
    $dec = if ($txt -match 'decode speed\s+([\d.]+) tok/s') { $Matches[1] } else { '' }
    $gen = if ($txt -match 'generated tokens\s+(\d+)')      { $Matches[1] } else { '' }
    "$($c.label),$rep,$dec,$gen"
  }
}
"== done =="
