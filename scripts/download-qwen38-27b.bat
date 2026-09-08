@echo off
setlocal enabledelayedexpansion

rem The Qwen3.8-27B dense artifact -- the default 27B for every benchmark in this repository, and
rem the one docs\config-calculator.html's "27b" rows are measured against.
rem
rem This script pinned the revision in its URL but verified nothing it received, and resumed
rem straight onto the final path: a truncated or corrupt 17 GB download was accepted silently. The
rem size and hash below are what HuggingFace reports for this revision. Structure matches
rem download-qwen36-27b.bat deliberately, so the two cannot drift.
set "REVISION=18dfc887423fa5aabf3cb56fac41490e462b3fab"
set "EXPECTED_SIZE=18210531328"
set "EXPECTED_SHA256=eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e"

set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
if defined NINFER_MODEL_DIR set "MODEL_DIR=%NINFER_MODEL_DIR%"
set "MODEL=%MODEL_DIR%\qwen3_8_27b.ninfer"

rem Staged under a revision-scoped name so that curl -C - can only ever resume the same artifact.
rem Resuming straight onto the final path appends at the current length without checking what
rem wrote those bytes, so a leftover partial from a different revision would be spliced into this
rem one and produce a plausibly sized, wholly corrupt file.
set "PART=%MODEL%.%REVISION%.part"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if exist "%MODEL%" (
  call :verify "%MODEL%"
  if "!VERIFY_OK!"=="1" (
    echo Model already present: %MODEL%
    exit /b 0
  )
  echo Existing %MODEL% did not verify against revision %REVISION%; fetching the pinned one.
)

echo Downloading the Qwen3.8-27B model (17.0 GiB)...
curl.exe -L -C - --fail --output "%PART%" "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/%REVISION%/qwen3_8_27b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)

call :verify "%PART%"
if not "!VERIFY_OK!"=="1" (
  echo Downloaded file at "%PART%" failed verification against revision %REVISION%. Delete it and run this file again.
  exit /b 1
)

move /y "%PART%" "%MODEL%" >nul
echo Model ready: %MODEL%
echo Point the tests at it with:  set NINFER_QWEN3_8_27B_WEIGHTS=%MODEL%
exit /b 0

rem Verifies %1 against EXPECTED_SIZE and, unless NINFER_SKIP_SHA256=1, EXPECTED_SHA256, setting
rem VERIFY_OK to 1 or 0. ACTUAL_SHA256 is cleared before the for /f loop: setlocal inherits
rem existing environment variables, so a stale value from outside this script would otherwise
rem survive "if not defined" and be compared unchanged.
:verify
set "VERIFY_OK=0"
set "VERIFY_PATH=%~1"
for %%A in ("%VERIFY_PATH%") do set "VERIFY_SIZE=%%~zA"
if not "!VERIFY_SIZE!"=="%EXPECTED_SIZE%" exit /b 0
if "%NINFER_SKIP_SHA256%"=="1" (
  set "VERIFY_OK=1"
  exit /b 0
)
set "ACTUAL_SHA256="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%VERIFY_PATH%" SHA256') do (
  if not defined ACTUAL_SHA256 set "ACTUAL_SHA256=%%H"
)
set "ACTUAL_SHA256=!ACTUAL_SHA256: =!"
if /i "!ACTUAL_SHA256!"=="%EXPECTED_SHA256%" set "VERIFY_OK=1"
exit /b 0
