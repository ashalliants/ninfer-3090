# Where does a decode step's time actually go?
#
# TODO.md section 2c says decode reaches only part of the card's memory bandwidth and that the
# next step is "a profile of one decode step to find where the stall is -- occupancy, L2
# behaviour, or a launch gap between the per-layer kernels". This is that profile.
#
# It answers the cheapest question first, which is the launch-gap one: add up the time the GPU
# spends inside kernels during the captured window and compare it with the wall time of the
# window. Whatever is missing is the GPU idle between kernels, and no amount of kernel tuning
# recovers it. Then it ranks kernels by total time so the next question has somewhere to go.
#
# ninfer_bench's --profile-measured brackets exactly one measured repetition with
# cudaProfilerStart/Stop, and nsys --capture-range=cudaProfilerApi records only that, so the
# report is one clean decode run rather than model loading and warmup.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\decode-step-profile.ps1
#
# A few minutes per configuration. Reports land under $NINFER_SWEEP_OUT as .nsys-rep plus the
# exported CSVs, which are the artifact -- the console summary below is a convenience.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

$modelDir = if ($env:NINFER_MODEL_DIR) { $env:NINFER_MODEL_DIR } else { 'C:\Ninefer-3090\models' }
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$nsys = if ($env:NINFER_NSYS) { $env:NINFER_NSYS } else {
  'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe' }
if (-not (Test-Path $nsys)) { throw "nsys not found at $nsys; set NINFER_NSYS" }

# Both models, because the two sit at very different fractions of the achievable ceiling and the
# reason is expected to differ: the dense one streams 15.7 GB per token in long contiguous runs,
# the MoE 2.5 GB of which 0.58 GB is a gather of 8 expert blocks out of 256 per layer.
$configs = @(
  @{ key='27b-dense'; path="$modelDir\qwen3_8_27b.ninfer";      depth=4096 }
  @{ key='35b-moe';   path="$modelDir\qwen3_6_35b_a3b.ninfer";  depth=4096 }
)

foreach ($c in $configs) {
  if (-not (Test-Path $c.path)) { "$($c.key): MISSING $($c.path)"; continue }
  $stem = "$out\prof_$($c.key)"
  Remove-Item "$stem.nsys-rep","$stem.sqlite" -ErrorAction SilentlyContinue

  & $nsys profile --force-overwrite true --output $stem `
      --trace cuda --capture-range cudaProfilerApi --capture-range-end stop `
      .\build-ninja\bench\ninfer_bench.exe `
      --weights $c.path --kv-dtype int8 --max-ctx 8192 `
      -pg "$($c.depth),128" -r 1 --warmup 1 --profile-measured `
      > "$stem.log" 2>&1
  if (-not (Test-Path "$stem.nsys-rep")) {
    "$($c.key): profile FAILED"
    Get-Content "$stem.log" -Tail 5 | ForEach-Object { "    $_" }
    continue
  }

  & $nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv `
      --output "$stem" "$stem.nsys-rep" > "$stem.stats.log" 2>&1

  $trace = "$stem`_cuda_gpu_trace.csv"
  $kern  = "$stem`_cuda_gpu_kern_sum.csv"
  if (-not (Test-Path $trace)) { "$($c.key): no trace CSV"; continue }

  # Busy time is the union of kernel intervals; wall time is first start to last end. The
  # difference is the GPU sitting idle waiting for the host to launch the next kernel, which on a
  # per-layer decode with dozens of small launches is the failure mode worth ruling out first.
  $rows = Import-Csv $trace | Where-Object { $_.'Start (ns)' -and $_.'Duration (ns)' }
  if (-not $rows) { "$($c.key): trace CSV had no kernel rows"; continue }
  $starts = $rows | ForEach-Object { [double]$_.'Start (ns)' }
  $ends   = $rows | ForEach-Object { [double]$_.'Start (ns)' + [double]$_.'Duration (ns)' }
  $busy   = ($rows | Measure-Object -Property 'Duration (ns)' -Sum).Sum
  $wall   = ($ends | Measure-Object -Maximum).Maximum - ($starts | Measure-Object -Minimum).Minimum

  ""
  "== $($c.key) at depth $($c.depth), int8, one measured repetition =="
  "  kernel launches        : {0:N0}" -f $rows.Count
  "  GPU busy               : {0:N3} ms" -f ($busy / 1e6)
  "  window wall            : {0:N3} ms" -f ($wall / 1e6)
  "  idle between kernels   : {0:N3} ms  ({1:N1}% of the window)" -f `
      (($wall - $busy) / 1e6), (100 * ($wall - $busy) / $wall)
  if (Test-Path $kern) {
    "  top kernels by total time:"
    Import-Csv $kern | Select-Object -First 8 | ForEach-Object {
      "    {0,8:N2} ms  {1,6} x  {2}" -f ([double]$_.'Total Time (ns)' / 1e6), $_.Instances, $_.Name
    }
  }
}
"== done =="
