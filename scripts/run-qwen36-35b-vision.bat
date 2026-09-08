@echo off
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%models\qwen3_6_35b_a3b.ninfer"
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
  echo Put this launcher beside the release files.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Run download-qwen36-35b-a3b.bat first, or drag the model onto this launcher.
  exit /b 1
)

echo Starting Qwen3.6-35B-A3B Vision at http://%HOST%:%PORT%/v1
echo Safe RTX 3090 profile: one request, 32K context, vision enabled, MTP disabled
"%SERVER%" "%MODEL%" --host %HOST% --port %PORT% --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-pending-requests 8 --pending-timeout-ms 600000 --prefill-chunk 512 --kv-dtype int8 --default-max-tokens 512 --vision --no-thinking --temperature 0.2
