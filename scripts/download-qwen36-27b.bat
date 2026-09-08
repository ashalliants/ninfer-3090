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
  for %%A in ("%MODEL%") do set "FOUND_SIZE=%%~zA"
  if "!FOUND_SIZE!"=="%EXPECTED_SIZE%" (
    echo Model already present: %MODEL%
    exit /b 0
  )
  echo Existing %MODEL% is not revision %REVISION%; fetching the pinned one.
)

echo Downloading the Qwen3.6-27B model (16.3 GiB)...
curl.exe -L -C - --fail --output "%PART%" "https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/%REVISION%/qwen3_6_27b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)

for %%A in ("%PART%") do set "ACTUAL_SIZE=%%~zA"
if not "!ACTUAL_SIZE!"=="%EXPECTED_SIZE%" (
  echo Expected %EXPECTED_SIZE% bytes, got !ACTUAL_SIZE!. Delete "%PART%" and run this file again.
  exit /b 1
)

rem Set NINFER_SKIP_SHA256=1 to skip this: it costs a full re-read of 16.3 GiB. The size check above
rem already rejects a truncated or spliced file, so this one is here for silent corruption.
if not "%NINFER_SKIP_SHA256%"=="1" (
  for /f "skip=1 delims=" %%H in ('certutil -hashfile "%PART%" SHA256') do (
    if not defined ACTUAL_SHA256 set "ACTUAL_SHA256=%%H"
  )
  set "ACTUAL_SHA256=!ACTUAL_SHA256: =!"
  if /i not "!ACTUAL_SHA256!"=="%EXPECTED_SHA256%" (
    echo Checksum mismatch ^(expected %EXPECTED_SHA256%, got !ACTUAL_SHA256!^).
    echo Delete "%PART%" and run this file again.
    exit /b 1
  )
)

move /y "%PART%" "%MODEL%" >nul
echo Model ready: %MODEL%
echo Point the tests at it with:  set NINFER_QWEN3_6_27B_WEIGHTS=%MODEL%
