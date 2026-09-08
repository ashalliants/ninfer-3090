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
  call :verify "%MODEL%"
  if "!VERIFY_OK!"=="1" (
    echo Model already present: %MODEL%
    exit /b 0
  )
  echo Existing %MODEL% did not verify against revision %REVISION%; fetching the pinned one.
)

echo Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model (21.2 GiB)...
curl.exe -L -C - --fail --output "%PART%" "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/%REVISION%/qwen3_6_35b_a3b.ninfer"
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
