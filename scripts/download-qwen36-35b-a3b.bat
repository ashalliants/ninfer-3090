@echo off
setlocal enabledelayedexpansion

rem Revision 560f227e, not c8b8c1c0. The older pin predates the DFlash bundle: its
rem artifact-manifest.json has no "dflash" source at all, so an artifact fetched with it cannot run
rem --spec dflash and makes ninfer_qwen3_6_35b_a3b_dflash_load_plan_test skip with "this artifact
rem carries no DFlash bundle". 560f227e adds it (from z-lab/Qwen3.6-35B-A3B-DFlash) for 0.38 GB
rem more. If you repin this, check the manifest still lists a dflash source, and update the size
rem and checksum below along with it.
set "REVISION=560f227e5a7104756d1a108201a8aa75654ea688"
set "EXPECTED_SIZE=22783246080"
set "EXPECTED_SHA256=1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2"

set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
if defined NINFER_MODEL_DIR set "MODEL_DIR=%NINFER_MODEL_DIR%"
set "MODEL=%MODEL_DIR%\qwen3_6_35b_a3b.ninfer"

rem curl -C - resumes by appending at the current file length, without checking what wrote those
rem bytes. Because the pinned revision changed, a partial download of the *previous* artifact sits
rem at exactly this path on any machine that ran the older script, and resuming onto it would splice
rem the tail of one artifact onto the head of another: a file of entirely plausible size that is
rem corrupt throughout. Staging under a name that carries the revision means a resume can only ever
rem continue the same artifact, and the checks below are what promote it to the final name.
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

echo Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model (21.2 GiB)...
curl.exe -L -C - --fail --output "%PART%" "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/%REVISION%/qwen3_6_35b_a3b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)

for %%A in ("%PART%") do set "ACTUAL_SIZE=%%~zA"
if not "!ACTUAL_SIZE!"=="%EXPECTED_SIZE%" (
  echo Expected %EXPECTED_SIZE% bytes, got !ACTUAL_SIZE!. Delete "%PART%" and run this file again.
  exit /b 1
)

rem Set NINFER_SKIP_SHA256=1 to skip this: it costs a full re-read of 21.2 GiB. The size check above
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
