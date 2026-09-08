$ErrorActionPreference = 'Continue'
Set-Location 'C:\ninfer-fork\ninfer-3090'

# Model directory, overridable so these are not welded to one machine.
$modelDir = if ($env:NINFER_MODEL_DIR) { $env:NINFER_MODEL_DIR } else { 'C:\Ninefer-3090\models' }
$out = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { "profiles\sweeps" }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$models = @(
  @{ key='27b-dense'; path="$modelDir\qwen3_8_27b.ninfer" },
  @{ key='35b-moe';   path="$modelDir\qwen3_6_35b_a3b.ninfer" }
)
$dtypes = @('bf16','int8','fp8','rk8v4','k8v4','nvfp4')

# ---------------------------------------------------------------------------------------------
# 1. Automatic-sizing context per dtype. README publishes this for int8 and rk8v4 only; the engine
#    resolves it against a headroom boundary rather than a fixed byte budget, so it cannot be
#    derived from bytes-per-token and has to be read off a real load. No --max-ctx = auto.
# ---------------------------------------------------------------------------------------------
"== auto capacity =="
"model,kv,auto_max_context,kv_payload_bytes"
foreach ($m in $models) {
  foreach ($d in $dtypes) {
    $csv = "$out\auto_$($m.key)_$d.csv"
    & .\build-ninja\bench\ninfer_bench.exe --weights $m.path --kv-dtype $d `
        -n 1 -r 1 --warmup 0 -o csv --output-file $csv > "$out\auto_$($m.key)_$d.log" 2>&1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $csv)) {
      "$($m.key),$d,FAILED,"
      Get-Content "$out\auto_$($m.key)_$d.log" -Tail 2 | ForEach-Object { "    $_" }
      continue
    }
    $r = Import-Csv $csv | Select-Object -First 1
    "$($m.key),$d,$($r.max_context),$($r.kv_payload_bytes)"
  }
}

# ---------------------------------------------------------------------------------------------
# 2. Does carrying DFlash cost anything when DFlash is not selected? docs/rtx-3090-windows.md
#    recommends the older, 0.38 GiB smaller pin on the grounds that it is what qualified the 24 GB
#    profiles. If those bytes are only resident when --spec dflash is chosen, that objection is
#    moot and the DFlash-bearing revision is strictly the better default.
# ---------------------------------------------------------------------------------------------
"== dflash residency =="
"artifact,spec,weights_bytes,host_to_device_bytes"
$artifacts = @(
  @{ key='v1-no-dflash'; path="$modelDir\qwen3_6_35b_a3b_v1_no_dflash.ninfer" },
  @{ key='v2-dflash';    path="$modelDir\qwen3_6_35b_a3b.ninfer" }
)
foreach ($a in $artifacts) {
  if (-not (Test-Path $a.path)) { "$($a.key),,MISSING,"; continue }
  foreach ($s in @(@{l='none';a=@()}, @{l='dflash-3';a=@('--spec','dflash','--draft-tokens','3')})) {
    $csv = "$out\dfc_$($a.key)_$($s.l).csv"
    $argv = @('--weights',$a.path,'--kv-dtype','int8','--max-ctx','8192',
              '-n','1','-r','1','--warmup','0','-o','csv','--output-file',$csv) + $s.a
    & .\build-ninja\bench\ninfer_bench.exe @argv > "$out\dfc_$($a.key)_$($s.l).log" 2>&1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $csv)) {
      "$($a.key),$($s.l),FAILED,"
      Get-Content "$out\dfc_$($a.key)_$($s.l).log" -Tail 2 | ForEach-Object { "    $_" }
      continue
    }
    $r = Import-Csv $csv | Select-Object -First 1
    "$($a.key),$($s.l),$($r.weights_capacity_bytes),$($r.load_host_to_device_bytes)"
  }
}

# ---------------------------------------------------------------------------------------------
# 3. Confirm the per-token model has no fixed intercept. The speed sweeps all pin --max-ctx 40960,
#    so they show the slope at one point only; if sequence capacity has a constant term, every
#    context estimate in the calculator is biased by it.
# ---------------------------------------------------------------------------------------------
"== per-token linearity =="
"model,kv,max_ctx,kv_payload_bytes,sequence_bytes"
foreach ($m in $models) {
  foreach ($d in @('int8','nvfp4')) {
    foreach ($c in @(8192, 16384, 32768, 65536)) {
      $csv = "$out\mem_$($m.key)_$($d)_$c.csv"
      & .\build-ninja\bench\ninfer_bench.exe --weights $m.path --kv-dtype $d --max-ctx $c `
          -n 1 -r 1 --warmup 0 -o csv --output-file $csv > "$out\mem_$($m.key)_$($d)_$c.log" 2>&1
      if ($LASTEXITCODE -ne 0 -or -not (Test-Path $csv)) { "$($m.key),$d,$c,FAILED,"; continue }
      $r = Import-Csv $csv | Select-Object -First 1
      "$($m.key),$d,$c,$($r.kv_payload_bytes),$($r.sequence_capacity_bytes)"
    }
  }
}
"== done =="
