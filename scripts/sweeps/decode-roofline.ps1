$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
$d = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }

# Reads the per-run CSVs that kv-decode-vs-depth.ps1 leaves in $NINFER_SWEEP_OUT -- run that first.
# Pure arithmetic, no GPU needed.
#
# Note those CSVs are ninfer_bench's own full output, written by --output-file, and carry every
# column including weights_capacity_bytes. They are NOT the six-column summary that sweep prints to
# stdout; that summary is a convenience for reading progress and drops most of the fields. The
# files are the artifact. (kv-decode-vs-depth.ps1 writes "<model>_<kv>.csv"; this reads the same.)
#
# TWO denominators, because only one of them is a ceiling anything can actually reach.
#
# 936.1 GB/s is the 3090's advertised figure: 384-bit GDDR6X at 19.5 Gbps. No kernel reaches it --
# refresh, ECC-less but real bus turnaround, and request efficiency all cost something. Measured
# here with tools/hbm_bandwidth_probe.cu, 4 GiB working set (683x L2), best of five trials:
#
#     cudaMemsetAsync (write)   863.3 GB/s   92.2% of advertised
#     kernel uint4 read         854.2 GB/s   91.3%
#     kernel uint4 write        824.1 GB/s   88.0%
#     cudaMemcpyAsync D2D       813.2 GB/s   86.9%
#
# Decode streams weights and KV *in*, so the read rate is its ceiling: 854.2 GB/s. Reporting
# against the advertised number instead understates the decode path by about six points and makes
# the headroom look larger than it is. Both columns are printed so the difference stays visible.
#
# Reproduce with:
#   nvcc -O3 -std=c++17 -arch=sm_86 tools/hbm_bandwidth_probe.cu -o hbm_probe && ./hbm_probe
$ADVERTISED_GBs = 936.1
$ACHIEVABLE_GBs = 854.2   # measured sustained read, see above

"{0,-10} {1,-7} {2,-8} {3,-9} {4,-11} {5,-11} {6,-14} {7}" -f 'model','kv','depth','tok/s','weights GB','KV GB','% advertised','% achievable'
foreach ($model in @('27b-dense','35b-moe')) {
  foreach ($kv in @('int8','rk8v4','fp8','k8v4','nvfp4','bf16')) {
    $f = "$d\${model}_$kv.csv"
    if (-not (Test-Path $f)) { continue }
    foreach ($r in (Import-Csv $f | Where-Object { $_.kind -eq 'pp+tg' })) {
      $tok    = [double]$r.decode_output_tok_s_mean
      $wGB    = [double]$r.weights_capacity_bytes / 1e9
      $perTok = [double]$r.kv_payload_bytes / 40960          # bytes of KV per token
      $kvGB   = ($perTok * [double]$r.n_prompt) / 1e9        # KV actually attended at this depth
      $bw     = ($wGB + $kvGB) * $tok
      "{0,-10} {1,-7} {2,-8} {3,-9} {4,-11} {5,-11} {6,-14} {7}" -f $model, $kv, $r.n_prompt,
        ("{0:N2}" -f $tok), ("{0:N2}" -f $wGB), ("{0:N3}" -f $kvGB),
        ("{0:N1}%" -f (100*$bw/$ADVERTISED_GBs)),
        ("{0:N1}% ({1:N0} GB/s)" -f (100*$bw/$ACHIEVABLE_GBs), $bw)
    }
  }
}

"`n=== prefill, for contrast (compute-bound rather than bandwidth-bound) ==="
"{0,-10} {1,-7} {2,-8} {3}" -f 'model','kv','prompt','prefill tok/s'
foreach ($model in @('27b-dense','35b-moe')) {
  $f = "$d\${model}_int8.csv"
  if (-not (Test-Path $f)) { continue }
  foreach ($r in (Import-Csv $f | Where-Object { $_.kind -eq 'pp+tg' })) {
    "{0,-10} {1,-7} {2,-8} {3}" -f $model, 'int8', $r.n_prompt, ("{0:N1}" -f [double]$r.prefill_tok_s_mean)
  }
}
