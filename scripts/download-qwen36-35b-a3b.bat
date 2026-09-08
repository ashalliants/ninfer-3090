@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_6_35b_a3b.ninfer"

rem Revision 560f227e, not c8b8c1c0. The older pin predates the DFlash bundle: its
rem artifact-manifest.json has no "dflash" source at all, so an artifact fetched with it cannot run
rem --spec dflash and makes ninfer_qwen3_6_35b_a3b_dflash_load_plan_test skip with "this artifact
rem carries no DFlash bundle". 560f227e adds it (from z-lab/Qwen3.6-35B-A3B-DFlash) for 0.38 GB
rem more. If you repin this, check the manifest still lists a dflash source.
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
echo Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model...
curl.exe -L -C - --fail --output "%MODEL%" "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/560f227e5a7104756d1a108201a8aa75654ea688/qwen3_6_35b_a3b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)
echo Model ready: %MODEL%
