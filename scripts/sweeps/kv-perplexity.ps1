$ErrorActionPreference = 'Continue'
Set-Location 'C:\ninfer-fork\ninfer-3090'

# Model directory, overridable so these are not welded to one machine.
$modelDir = if ($env:NINFER_MODEL_DIR) { $env:NINFER_MODEL_DIR } else { 'C:\Ninefer-3090\models' }

# Same corpus, mode and window as the three already recorded in README.md, so the new rows are
# directly comparable rather than a separate experiment: ninfer-ppl-1m-v1, --quick, 4096/2048.
$dtypes = @('fp8','nvfp4','k8v4')
$out = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { "profiles\sweeps" }
New-Item -ItemType Directory -Force -Path $out | Out-Null

foreach ($d in $dtypes) {
    $log = "$out\$d.log"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & .\build-ninja\apps\ninfer-perplexity.exe `
        "$modelDir\qwen3_8_27b.ninfer" `
        --corpus 'eval\corpora\perplexity-1m\manifest.json' --quick `
        --context 4096 --stride 2048 --kv-dtype $d `
        --output "$out\$d" --log-level warning > $log 2>&1
    $code = $LASTEXITCODE
    $report = Get-ChildItem -Recurse -Filter report.json "$out\$d" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($report) {
        $j = Get-Content $report.FullName -Raw | ConvertFrom-Json
        "{0,-7} ppl={1}  tokens={2}  {3}s" -f $d, $j.overall.perplexity, $j.overall.scored_tokens, [math]::Round($sw.Elapsed.TotalSeconds,0)
    } else {
        "{0,-7} FAILED (exit {1})" -f $d, $code
        Get-Content $log -Tail 3 | ForEach-Object { "        $_" }
    }
}
