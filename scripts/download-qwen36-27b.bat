@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_6_27b.ninfer"

rem The Qwen3.6 27B groupwise-int artifact, target_key qwen3_6_27b. This is a different model
rem family from qwen3_8_27b, which is why having the latter does not satisfy the former: four
rem real-model tests -- ninfer_qwen3_6_27b_prefix_real_test, _score_real_test, _load_plan_test and
rem the Qwen3.6 27B half of the engine suite -- skip without it.
rem
rem Pinned at faaa0c14 (17,495,365,888 bytes,
rem sha256 7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b).

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
echo Downloading the Qwen3.6-27B model (16.3 GB)...
curl.exe -L -C - --fail --output "%MODEL%" "https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/faaa0c140d0a92743872256a8b78a954b3984018/qwen3_6_27b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)
echo Model ready: %MODEL%
echo Point the tests at it with:  set NINFER_QWEN3_6_27B_WEIGHTS=%MODEL%
