@echo off
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%models\qwen3_8_27b.ninfer"
rem Overridable without editing this file:  set NINFER_HOST=0.0.0.0 && this-script.bat
rem Loopback by default -- 0.0.0.0 publishes an unauthenticated OpenAI-compatible endpoint to every
rem network this machine is on, so it is opt-in per run rather than the shipped default.
set "HOST=127.0.0.1"
set "PORT=8080"
if not "%NINFER_SERVER%"=="" set "SERVER=%NINFER_SERVER%"
if not "%NINFER_MODEL%"=="" set "MODEL=%NINFER_MODEL%"
if not "%NINFER_HOST%"=="" set "HOST=%NINFER_HOST%"
if not "%NINFER_PORT%"=="" set "PORT=%NINFER_PORT%"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Put this launcher beside the v0.6.1 release files.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Run download-qwen38-27b.bat first, or drag a qwen3_8_27b.ninfer file onto this launcher.
  exit /b 1
)

echo Starting Qwen3.8-27B at http://%HOST%:%PORT%/v1
echo Profile: up to eight requests, 8K context, MTP3, ReplaySSM
"%SERVER%" "%MODEL%" --host %HOST% --port %PORT% --max-context 8192 --kv-capacity 16384 --max-concurrency 8 --max-pending-requests 32 --pending-timeout-ms 600000 --prefill-chunk 512 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft
