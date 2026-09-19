@echo off
setlocal

rem ======================== EDITABLE SETTINGS ========================
set "MAX_CONTEXT=131072"
set "OUTPUT_TOKENS=1024"
set "PREFILL_PROMPT_CHARACTERS=28000"
set "COHORTS=1,2,4,8"
set "KV_DTYPE=rk8v4"
REM Hands wide prefill GEMMs to cuBLAS: about 1.73x prefill for +0.156%% perplexity (4.343155 ->
REM 4.349944 on the 1M corpus). Set to 0 to measure the default-quality engine instead. The chunk
REM follows it, because the route only amortises its weight-sized dequantise over a call's tokens.
REM DFlash2 at K=7 measured 172.3 tok/s against MTP3's 124.3 on realistic generation, so 1.39x.
REM It needs the artifact carrying the draft model. K is workload-dependent: K=15 wins on the
REM synthetic corpus and loses on real generation, so tune it against the workload being sold.
set "SPEC=dflash2"
set "DRAFT_TOKENS="
set "PREFILL_CUBLAS=1"
set "PREFILL_CHUNK="
set "START_DELAY_SECONDS=10"
set "MODEL=%~dp0..\..\qwen3_8_27b.ninfer"
set "SERVER=%~dp0..\build-ninja\apps\ninfer-serve.exe"
rem ==================================================================

for %%I in ("%~dp0..") do set "REPO=%%~fI"
if not exist "%SERVER%" (
  echo ERROR: Server not found: %SERVER%
  exit /b 1
)
if not exist "%MODEL%" (
  echo ERROR: Model not found: %MODEL%
  exit /b 1
)
where uv >nul 2>nul || (
  echo ERROR: uv is not available in PATH.
  exit /b 1
)

echo.
echo RTX 3090 Qwen3.8 benchmark
echo   Shared context : %MAX_CONTEXT% tokens ^(C1 full; C8 capped at 8K per request^)
echo   Decode output  : %OUTPUT_TOKENS% tokens
echo   Cohorts        : C1, C2, C4, C8
echo   KV cache       : %KV_DTYPE%
echo   Speculation    : %SPEC% %DRAFT_TOKENS%
if "%PREFILL_CUBLAS%"=="0" (echo   Prefill route  : default integer-activation) else (echo   Prefill route  : cuBLAS, +0.156%% perplexity ^(set PREFILL_CUBLAS=0 for the default engine^))
echo   Results        : %REPO%\benchmark_results\windows_3090_*
if /I "%KV_DTYPE%"=="int8" if %MAX_CONTEXT% GTR 65536 echo WARNING: This high-context INT8 profile is not the recommended 3090 benchmark setting.
echo.
echo Starting in %START_DELAY_SECONDS% seconds. Press Ctrl+C to cancel.
timeout /t %START_DELAY_SECONDS% /nobreak >nul

set "NINFER_BENCH_SERVER=%SERVER%"
set "NINFER_BENCH_MODEL=%MODEL%"
set "NINFER_BENCH_MAX_CONTEXT=%MAX_CONTEXT%"
set "NINFER_BENCH_OUTPUT_TOKENS=%OUTPUT_TOKENS%"
set "NINFER_BENCH_PREFILL_CHARS=%PREFILL_PROMPT_CHARACTERS%"
set "NINFER_BENCH_COHORTS=%COHORTS%"
set "NINFER_BENCH_KV_DTYPE=%KV_DTYPE%"
set "NINFER_BENCH_SPEC=%SPEC%"
if not "%DRAFT_TOKENS%"=="" set "NINFER_BENCH_DRAFT_TOKENS=%DRAFT_TOKENS%"
set "NINFER_BENCH_PREFILL_CUBLAS=%PREFILL_CUBLAS%"
if not "%PREFILL_CHUNK%"=="" set "NINFER_BENCH_PREFILL_CHUNK=%PREFILL_CHUNK%"

pushd "%REPO%"
uv run tools\bench\run_qwen38_windows_3090_benchmarks.py
set "RESULT=%ERRORLEVEL%"
popd

if not "%RESULT%"=="0" (
  echo.
  echo BENCHMARK FAILED. Logs and partial results were preserved.
  exit /b %RESULT%
)
echo.
echo BENCHMARK COMPLETE. Open the results directory printed above.
exit /b 0
