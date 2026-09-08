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
# RTX 3090: 384-bit GDDR6X at 19.5 Gbps = 936.2 GB/s theoretical. Decode is memory-bound: every
# token streams the resident weights once, plus the KV it attends over. So achieved bandwidth
# divided by peak says how much of the card the decode path is actually using -- the single number
# that says whether there is headroom left, independent of any kernel detail.
$PEAK_GBs = 936.2

"{0,-10} {1,-7} {2,-8} {3,-9} {4,-11} {5,-11} {6}" -f 'model','kv','depth','tok/s','weights GB','KV GB','% of peak BW'
foreach ($model in @('27b-dense','35b-moe')) {
  foreach ($kv in @('int8','rk8v4')) {
    $f = "$d\${model}_$kv.csv"
    if (-not (Test-Path $f)) { continue }
    foreach ($r in (Import-Csv $f | Where-Object { $_.kind -eq 'pp+tg' })) {
      $tok    = [double]$r.decode_output_tok_s_mean
      $wGB    = [double]$r.weights_capacity_bytes / 1e9
      $perTok = [double]$r.kv_payload_bytes / 40960          # bytes of KV per token
      $kvGB   = ($perTok * [double]$r.n_prompt) / 1e9        # KV actually attended at this depth
      $bw     = ($wGB + $kvGB) * $tok
      "{0,-10} {1,-7} {2,-8} {3,-9} {4,-11} {5,-11} {6}" -f $model, $kv, $r.n_prompt,
        ("{0:N2}" -f $tok), ("{0:N2}" -f $wGB), ("{0:N3}" -f $kvGB),
        ("{0:N1}% ({1:N0} GB/s)" -f (100*$bw/$PEAK_GBs), $bw)
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
