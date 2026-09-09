@echo off
setlocal enabledelayedexpansion

rem The Qwen3.6 27B groupwise-int artifact, target_key qwen3_6_27b. This is a different model
rem family from qwen3_8_27b, which is why having the latter does not satisfy the former: four
rem real-model tests -- ninfer_qwen3_6_27b_prefix_real_test, _score_real_test, _load_plan_test and
rem the Qwen3.6 27B half of the engine suite -- skip without it.
set "REVISION=faaa0c140d0a92743872256a8b78a954b3984018"
set "EXPECTED_SIZE=17495365888"
set "EXPECTED_SHA256=7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b"

set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
if defined NINFER_MODEL_DIR set "MODEL_DIR=%NINFER_MODEL_DIR%"
set "MODEL=%MODEL_DIR%\qwen3_6_27b.ninfer"

rem Staged under a revision-scoped name so that curl -C - can only ever resume the same artifact.
rem Resuming straight onto the final path appends at the current length without checking what wrote
rem those bytes, so a leftover partial from a different revision would be spliced into this one and
rem produce a plausibly sized, wholly corrupt file. See download-qwen36-35b-a3b.bat, where changing
rem the pin made that a live hazard rather than a hypothetical one.
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

echo Downloading the Qwen3.6-27B model (16.3 GiB)...
curl.exe -L -C - --fail --output "%PART%" "https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/%REVISION%/qwen3_6_27b.ninfer"
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
if errorlevel 1 (
  echo Failed to move "%PART%" to "%MODEL%". Delete "%PART%" and run this file again.
  exit /b 1
)
echo Model ready: %MODEL%
echo Point the tests at it with:  set NINFER_QWEN3_6_27B_WEIGHTS=%MODEL%
exit /b 0

rem Verifies %1 against EXPECTED_SIZE and, unless NINFER_SKIP_SHA256=1, EXPECTED_SHA256, setting
rem VERIFY_OK to 1 or 0. Used both for an existing MODEL (so a same-sized-but-corrupt file is not
rem accepted forever just because it happened to pass once, or was replaced out from under this
rem script) and for a freshly downloaded PART -- one check that cannot drift out of sync with
rem itself. ACTUAL_SHA256 is cleared before the for /f loop below: setlocal inherits existing
rem environment variables, so a stale ACTUAL_SHA256 left over from outside this script would
rem otherwise survive "if not defined" and be compared unchanged.
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
