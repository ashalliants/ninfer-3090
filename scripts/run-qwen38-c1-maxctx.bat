@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Qwen3.8-27B on one RTX 3090, tuned for context and prefix reuse rather than for caution.
rem
rem run-qwen38-c1.bat serves 65,536 tokens of INT8 and leaves 2.85 GiB of the card unused. This
rem profile spends it: rk8v4 KV, MTP3 speculation plus the draft head, and the tuned context cache.
rem
rem WHAT rk8v4 BUYS. The same KV that holds 171,648 INT8 tokens holds 226,560 rk8v4 tokens -- +33%
rem context for +0.082%% perplexity. It is opt-in precisely because INT8 is the quality default.
rem
rem CONTEXT CACHE. A checkpoint is a KV prefix plus a StateImage, and on this model the StateImage
rem is 147 MiB flat regardless of prefix length -- 48 GDN layers of 128x128x48 FP32 recurrent state
rem plus conv. That is 2.4x the 35B-A3B's 61.4 MiB, so the slots are correspondingly expensive:
rem --host-state-slots 32 pins 4.59 GiB of HOST memory, not device. It is what takes prefix reuse
rem from 8.4%% to 98.3%% on a multi-preamble workload.
rem
rem --auto-prefix-grid lets two callers whose prompts merely start alike share a cached prefix with
rem no client hint. A grid point is only materialised once two independent callers have both asked
rem for it, so it cannot waste a slot speculatively.
rem
rem MEASURED on this machine with the desktop running, which is the pessimistic case:
rem
rem   lanes  KV      context   vision   runtime    free after startup
rem   ------------------------------------------------------------------
rem   1      int8     65,536   off      2.73 GiB   2.85 GiB   <- what run-qwen38-c1.bat does
rem   1      rk8v4   131,072   off      3.93 GiB   1.68 GiB
rem   1      rk8v4   131,072   overlay  3.94 GiB   1.59 GiB   <- default here
rem   2      rk8v4   131,072   overlay  4.32 GiB   1.26 GiB
rem   1      rk8v4   163,840   overlay  4.78 GiB   763.2 MiB
rem
rem Windows keeps one lane by default: a desktop holds roughly 1.5 GiB of the card, so the
rem headroom above is what you actually have. Vision is on -- overlay residency costs about 10 MiB
rem of runtime reservation, so there is no reason to trade it away. run-qwen38-vision.bat remains
rem for the plain 32K image profile.
rem Rungs if startup refuses: 163840 / 131072 / 114688 / 98304 / 65536.
rem ---------------------------------------------------------------------------------------------

set "MODEL=%~dp0..\..\qwen3_8_27b.ninfer"
if not "%NINFER_MODEL%"=="" set "MODEL=%NINFER_MODEL%"

set "CONTEXT=131072"
set "CONCURRENCY=1"
set "HOST=127.0.0.1"
set "PORT=8080"

set "ROOT=%~dp0.."
set "SERVER=%ROOT%\build-ninja\apps\ninfer-serve.exe"
if not exist "%SERVER%" set "SERVER=%~dp0ninfer-serve.exe"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Build it first:  .\scripts\build.ps1
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Download it first:  download-qwen38-27b.bat
  exit /b 1
)

echo Qwen3.8-27B  ^|  C%CONCURRENCY%  ^|  context %CONTEXT%  ^|  rk8v4 KV  ^|  MTP3 + draft head  ^|  Vision (overlay)
echo Cache: 8 shared / 8 private / 32 host states  ^|  automatic prefix grid on
echo API: http://%HOST%:%PORT%/v1
echo.

"%SERVER%" "%MODEL%" ^
  --host %HOST% --port %PORT% ^
  --max-concurrency %CONCURRENCY% ^
  --max-context %CONTEXT% ^
  --kv-capacity %CONTEXT% ^
  --kv-dtype rk8v4 ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --prefill-chunk 1024 ^
  --max-pending-requests 16 --pending-timeout-ms 600000 ^
  --vision --vision-residency overlay ^
  --max-private-continuations 8 --max-shared-prefixes 8 --host-state-slots 32 --host-kv-mib 8192 ^
  --auto-prefix-grid

endlocal
