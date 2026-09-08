$ErrorActionPreference = 'Continue'
# Run from the repository root regardless of where this is invoked from, rather than a hardcoded
# path: these are committed, and the next person's checkout will not be at C:\ninfer-fork.
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

# Both overridable so this is not welded to one machine. The output directory is created here
# because it does not exist in a clean checkout -- profiles/ is gitignored.
$modelDir = if ($env:NINFER_MODEL_DIR) { $env:NINFER_MODEL_DIR } else { 'C:\Ninefer-3090\models' }
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$mdir = $modelDir

# The loader rejects anything not named *.ninfer, which is why the first attempt at the DFlash
# residency probe failed on the .pre-dflash.bak backup. Rename rather than copy: this artifact is
# 20.84 GB and a copy would cost real disk for no reason. The new name is also clearer than a .bak
# suffix about what the file actually is.
$oldBak = "$mdir\qwen3_6_35b_a3b.ninfer.pre-dflash.bak"
$v1     = "$mdir\qwen3_6_35b_a3b_v1_no_dflash.ninfer"
if ((Test-Path $oldBak) -and -not (Test-Path $v1)) { Move-Item -LiteralPath $oldBak -Destination $v1 }
"v1 artifact present: $(Test-Path $v1)"

"== dflash residency =="
"artifact,spec,weights_bytes,host_to_device_bytes,artifact_bytes"
foreach ($a in @(@{k='v1-no-dflash'; p=$v1},
                 @{k='v2-dflash';    p="$mdir\qwen3_6_35b_a3b.ninfer"})) {
  if (-not (Test-Path $a.p)) { "$($a.k),,MISSING,,"; continue }
  $sz = (Get-Item $a.p).Length
  foreach ($s in @(@{l='none';a=@()}, @{l='dflash-3';a=@('--spec','dflash','--draft-tokens','3')})) {
    $csv = "$out\dfc2_$($a.k)_$($s.l).csv"
    $argv = @('--weights',$a.p,'--kv-dtype','int8','--max-ctx','8192',
              '-n','1','-r','1','--warmup','0','-o','csv','--output-file',$csv) + $s.a
    & .\build-ninja\bench\ninfer_bench.exe @argv > "$out\dfc2_$($a.k)_$($s.l).log" 2>&1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $csv)) {
      "$($a.k),$($s.l),FAILED,,$sz"
      Get-Content "$out\dfc2_$($a.k)_$($s.l).log" -Tail 2 | ForEach-Object { "    $_" }
      continue
    }
    $r = Import-Csv $csv | Select-Object -First 1
    "$($a.k),$($s.l),$($r.weights_capacity_bytes),$($r.load_host_to_device_bytes),$sz"
  }
}

# Automatic-sizing context per format, via the serving path rather than the bench harness.
"== auto capacity =="
"model,kv,kv_capacity_line"
foreach ($m in @(@{k='27b-dense'; p="$mdir\qwen3_8_27b.ninfer"},
                 @{k='35b-moe';   p="$mdir\qwen3_6_35b_a3b.ninfer"})) {
  foreach ($d in @('bf16','int8','fp8','rk8v4','k8v4','nvfp4')) {
    $log = "$out\cap_$($m.k)_$d.log"
    & .\build-ninja\apps\ninfer.exe $m.p --prompt hi --max-new 1 --greedy `
        --kv-dtype $d --max-context 262144 --kv-capacity auto > $log 2>&1
    $code = $LASTEXITCODE
    $txt = Get-Content $log -Raw
    $cap = if ($txt -match '(?m)^.*KV capacity.*$') { $Matches[0].Trim() }
           else { "exit=$code no-kv-line" }
    "$($m.k),$d,$cap"
  }
}
"== done =="
