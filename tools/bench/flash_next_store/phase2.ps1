param([string[]]$Runs = @('nvme', 'store-ram', 'store-nvme', 'vision'))
$b   = 'C:\ninfer-fork\ninfer-3090\.claude\worktrees\linked-baking-piglet\.claude\worktrees\fn-store\build-ninja'
$exe = Join-Path $b 'tests\ninfer_tests.exe'
$s   = 'C:\Users\Ash\AppData\Local\Temp\claude\C--ninfer-fork-ninfer-3090\873ac902-dfeb-402f-a2b0-d463fb8eda65\scratchpad\fnstore'
$env:NINFER_TEST_QWEN4EXP_ARTIFACT = 'C:\ninfer-fork\ninfer-3090\models\qwen4exp-iq2xs-mtp-vision.ninfer'
Set-Location $b
function Fresh([string]$name) { $d = Join-Path $s $name; if (Test-Path $d) { Remove-Item -Recurse -Force $d }; $d }
foreach ($r in $Runs) {
  $ram = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
  "=== $r (free RAM $ram GiB) $(Get-Date -Format o)"
  switch ($r) {
    'nvme'       { & $exe ninfer_qwen4_exp_engine_real_test --cache --mtp 3 --host-mib 0 --nvme-mib 8192 2>&1 }
    'nvme-plain' { & $exe ninfer_qwen4_exp_engine_real_test --cache --host-mib 0 --nvme-mib 8192 2>&1 }
    'ram'        { & $exe ninfer_qwen4_exp_engine_real_test --cache --mtp 3 2>&1 }
    'store-ram'  { & $exe ninfer_qwen4_exp_engine_real_test --cache --mtp 3 --store (Fresh 'store-ram') 2>&1 }
    'store-nvme' { & $exe ninfer_qwen4_exp_engine_real_test --cache --mtp 3 --host-mib 0 --nvme-mib 8192 --store (Fresh 'store-nvme') 2>&1 }
    'store-plain' { & $exe ninfer_qwen4_exp_engine_real_test --cache --host-mib 0 --nvme-mib 8192 --store (Fresh 'store-plain') 2>&1 }
    'vision'     { & $exe ninfer_qwen4_exp_vision_engine_real_test --cache --host-mib 0 --nvme-mib 8192 --store (Fresh 'store-vision') 2>&1 }
    'plain'      { & $exe ninfer_qwen4_exp_engine_real_test --cache --generate 32 2>&1 }
    'evict-ram'  { & $exe ninfer_qwen4_exp_engine_real_test --cache --evict --host-mib 512 --generate 32 2>&1 }
    'vision-ram' { & $exe ninfer_qwen4_exp_vision_engine_real_test --cache 2>&1 }
    'evict-nvme' { & $exe ninfer_qwen4_exp_engine_real_test --cache --evict --host-mib 0 --nvme-mib 1024 2>&1 }
  }
  "=== $r exit $LASTEXITCODE"
}
