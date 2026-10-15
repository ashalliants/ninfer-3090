@echo off
rem Where this script sits is read before delayed expansion is on, so an exclamation mark in the
rem install path survives. Everything below reads folders with !VAR!, never inside a ( ) block.
setlocal disabledelayedexpansion
set "ROOT=%~dp0"
for %%I in ("%ROOT%..") do set "REPO_ROOT=%%~fI"
for %%I in ("%ROOT%.") do set "SCRIPT_DIRNAME=%%~nxI"
setlocal enabledelayedexpansion

rem Downloads one pinned model artifact, verifies it, and only then promotes it to its final name.
rem
rem   download-model.bat <model>
rem
rem   qwen38-27b       Qwen3.8-27B, 17.7 GiB. The default 27B for every benchmark in this repository.
rem                    This fork's own conversion (WarlaxZ/Qwen3.8-27B-NInfer-3090, recipe
rem                    qwen3_8_27b): the imatrix-weighted grouped_search encoder, a 4-bit embedding, a
rem                    6-bit head and 4-bit mixer outputs and MLP down in layers 36-63. It carries the
rem                    vision, MTP and DFlash2 weights, so one file serves --vision, --spec mtp and
rem                    --spec dflash2. Executables older than the 5120x17408 Q4 linear_add route refuse
rem                    it at startup. Figures published before 2026-10 were measured against the
rem                    upstream neroued pin 1cbd84e7 (v2 18dfc887), 19.0 GiB.
rem   qwen36-27b       Qwen3.6-27B groupwise-int, 16.3 GiB, target_key qwen3_6_27b. A different model
rem                    family from qwen3_8_27b, which is why having the latter does not satisfy the
rem                    former: four real-model tests -- ninfer_qwen3_6_27b_prefix_real_test,
rem                    _score_real_test, _load_plan_test and the Qwen3.6 27B half of the engine suite
rem                    -- skip without it.
rem   qwen36-35b-a3b   Qwen3.6-35B-A3B, 21.2 GiB, the RTX 3090-compatible vision model. The official
rem                    v3 artifact revision: its artifact-manifest.json lists the text, vision, mtp
rem                    and dflash components, and the DFlash bundle (from
rem                    z-lab/Qwen3.6-35B-A3B-DFlash) is what --spec dflash needs.
rem   qwen38-flash-next  Qwen3.8-Flash-Next (preview, Windows), this fork's exact conversion of the
rem                    ISTA DASLab GSQ-RCO IQ2_XS GGUF with Strata's MTP drafter and the vision tower
rem                    (WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090). Three things are fetched: the model
rem                    file, its .part-0001 continuation (both read once, at load) and the 27.1 GiB
rem                    n-gram volume, which is read on every prompt. The script asks where the volume
rem                    goes (default: beside the model); a different folder is for a faster drive,
rem                    since prefill from an NVMe drive measured 1.1-1.4x that from a SATA SSD. The
rem                    launcher is then told with NINFER_NGRAM_VOLUME, which the script prints.
rem
rem To repin a model, edit its block below: take the size and sha256 from what HuggingFace reports for
rem the new revision (X-Linked-Size and X-Linked-ETag on the resolve URL, or the revision's
rem artifact-manifest.json), and for qwen36-35b-a3b check the manifest still lists a dflash component.
rem Changing a pin makes the staging hazard below live rather than hypothetical.
rem
rem Run with no argument (for instance by double-clicking it) it shows a numbered menu of the
rem models instead; an unknown name still prints the usage and exits 2. download-model.sh has no
rem menu and always requires the name.
rem
rem Environment: NINFER_MODEL_DIR (default: models\ at the repo root in a checkout, or beside this
rem file in the release archive), NINFER_SKIP_SHA256=1 to accept a file on size alone. qwen38-27b also
rem fetches the private godmode graft when HF_TOKEN, or `hf auth login`, grants access to it.
rem qwen38-flash-next: NINFER_NGRAM_DIR is the folder for the n-gram volume, which skips the question.
rem
rem Uses aria2c (16 parallel ranges) when it is on PATH, since single-stream curl against these
rem artifacts has been observed to throttle to ~1 MB/s or stall outright; falls back to curl
rem otherwise. Install it with `winget install aria2.aria2` or `choco install aria2` to opt in.

rem Only qwen38-flash-next has continuation parts and an n-gram volume. Cleared here because setlocal
rem inherits the caller's environment.
set "PART_COUNT=0"
set "NGRAM_NAME="

if "%~1"=="" goto :menu
if /i "%~1"=="qwen38-27b" goto :model_qwen38_27b
if /i "%~1"=="bonsai2-27b" goto :model_bonsai2_27b
if /i "%~1"=="qwen36-27b" goto :model_qwen36_27b
if /i "%~1"=="qwen36-35b-a3b" goto :model_qwen36_35b_a3b
if /i "%~1"=="qwen38-flash-next" goto :model_qwen38_flash_next
if /i "%~1"=="-h" goto :usage_help
if /i "%~1"=="--help" goto :usage_help
goto :usage_error

:menu
echo Which model do you want to download?
echo.
echo   1 = Qwen3.8-27B (recommended)       qwen38-27b
echo   2 = Qwen3.6-35B-A3B                 qwen36-35b-a3b
echo   3 = Qwen3.6-27B                     qwen36-27b
echo   4 = Qwen3.8-Flash-Next (preview)    qwen38-flash-next   (needs 64 GB of RAM)
echo   5 = Ternary Bonsai 2 27B            bonsai2-27b
echo.
choice /c 12345 /n /m "Choose 1-5: "
rem choice reports 255 when it cannot read a key (no console, closed stdin). Refuse rather than
rem let that fall through and pick a model nobody chose.
if errorlevel 255 exit /b 2
if errorlevel 5 goto :model_bonsai2_27b
if errorlevel 4 goto :model_qwen38_flash_next
if errorlevel 3 goto :model_qwen36_27b
if errorlevel 2 goto :model_qwen36_35b_a3b
if errorlevel 1 goto :model_qwen38_27b
exit /b 2

:usage_help
call :usage
exit /b 0

:usage_error
call :usage
exit /b 2

:usage
echo usage: %~nx0 ^<model^>
echo.
echo   qwen38-27b (recommended)
echo   bonsai2-27b
echo   qwen36-27b
echo   qwen36-35b-a3b
echo   qwen38-flash-next (preview)
exit /b 0

:model_qwen38_27b
set "ARTIFACT=qwen3_8_27b.ninfer"
set "OWNER=WarlaxZ"
set "REPO=Qwen3.8-27B-NInfer-3090"
set "REVISION=d47f2732d369acaec76dc44228f67c72b081df2f"
set "EXPECTED_SIZE=18982458624"
set "EXPECTED_SHA256=7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706"
set "LABEL=the Qwen3.8-27B model (17.7 GiB)"
set "TESTS_VARIABLE=NINFER_QWEN3_8_27B_WEIGHTS"
goto :model_selected

:model_bonsai2_27b
set "ARTIFACT=bonsai2_27b.ninfer"
set "OWNER=WarlaxZ"
set "REPO=Ternary-Bonsai-2-27B-NInfer-3090"
set "REVISION=983a9be22f3623f3a2799655741d2a92db86d37d"
set "EXPECTED_SIZE=9520051456"
set "EXPECTED_SHA256=975126774d7e1f9ffa36b654c0a6fc0db42c773ba4dd5f1196a9ddc6a46d4634"
set "LABEL=the Ternary Bonsai 2 27B model (8.9 GiB)"
set "TESTS_VARIABLE=NINFER_TEST_ARTIFACT"
goto :model_selected

:model_qwen36_27b
set "ARTIFACT=qwen3_6_27b.ninfer"
set "OWNER=neroued"
set "REPO=Qwen3.6-27B-NInfer"
set "REVISION=3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829"
set "EXPECTED_SIZE=17495538688"
set "EXPECTED_SHA256=9b610a7d051e7c4dbf89adb604bd269d248b643c8f6c7ef75bdb871af92c6f6b"
set "LABEL=the Qwen3.6-27B model (16.3 GiB)"
set "TESTS_VARIABLE=NINFER_QWEN3_6_27B_WEIGHTS"
goto :model_selected

:model_qwen36_35b_a3b
set "ARTIFACT=qwen3_6_35b_a3b.ninfer"
set "OWNER=neroued"
set "REPO=Qwen3.6-35B-A3B-NInfer"
set "REVISION=ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6"
set "EXPECTED_SIZE=22790484480"
set "EXPECTED_SHA256=3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84"
set "LABEL=the RTX 3090-compatible Qwen3.6-35B-A3B vision model (21.2 GiB)"
set "TESTS_VARIABLE=NINFER_QWEN3_6_35B_A3B_WEIGHTS"
goto :model_selected

rem The n-gram volume is pinned to the commit that uploaded it; the model file and its part are
rem pinned to the revision that published the model card with them. The volume is unchanged there.
:model_qwen38_flash_next
set "ARTIFACT=qwen4exp-iq2xs-mtp-vision.ninfer"
set "OWNER=WarlaxZ"
set "REPO=Qwen3.8-Flash-Next-NInfer-3090"
set "REVISION=c7dfb9694fc3cb3d40298afbfe2e3a8c172241fa"
set "EXPECTED_SIZE=32000000000"
set "EXPECTED_SHA256=87c871e6cc1b6392e387104e8dfb159283a8a64247f669fbf33355b19b769ef4"
set "PART_COUNT=1"
set "PART_1_SIZE=9012140032"
set "PART_1_SHA256=1798c63c2f74a05cac71a609701f34368f0ba1e1eeebbf1b7641e83d0dd8d811"
set "NGRAM_NAME=qwen4exp-iq2xs-mtp-vision.ninfer.ngram"
set "NGRAM_REVISION=95b1b0169bd25db96ceb772a8a67cc6e87088a1f"
set "NGRAM_SIZE=29127258112"
set "NGRAM_SHA256=7623459b2385eef4505891c2ff133b3dd8ad4ecd0ad334c901878f3fa1d1b051"
set "LABEL=the Qwen3.8-Flash-Next model (38.2 GiB in two files)"
set "TESTS_VARIABLE=NINFER_TEST_QWEN4EXP_ARTIFACT"
goto :model_selected

:model_selected
rem Two layouts share this script. In the release archive it sits at the archive root beside
rem models\. In a checkout it sits in scripts\, one level under the repo root, and models\ (see
rem .gitignore) is beside the repo root, not beside this script -- a directory named scripts with a
rem CMakeLists.txt above it tells the two apart (a bare CMakeLists.txt probe would misfire on an
rem archive unpacked beneath any source tree), matching run.bat's own default so a model
rem downloaded here is found there.
rem ROOT, REPO_ROOT and SCRIPT_DIRNAME were set at the top.
set "MODEL_DIR=!ROOT!models"
if /i "!SCRIPT_DIRNAME!"=="scripts" if exist "!REPO_ROOT!\CMakeLists.txt" set "MODEL_DIR=!REPO_ROOT!\models"
if defined NINFER_MODEL_DIR set "MODEL_DIR=!NINFER_MODEL_DIR!"
set "MODEL_DIR_FULL="
call :dir_check MODEL_DIR
if "!DIR_BAD!"=="1" goto :model_dir_rejected
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
for %%I in ("%MODEL_DIR%") do set "MODEL_DIR_FULL=%%~fI"
call :trim_trailing_backslash MODEL_DIR_FULL
rem From here on the model folder is the full path, so D: means the folder it resolved to.
set "MODEL_DIR=!MODEL_DIR_FULL!"
call :join_path MODEL MODEL_DIR "%ARTIFACT%"

rem Ask where the n-gram volume goes before anything is fetched, so the downloads can run unattended.
if defined NGRAM_NAME call :choose_ngram_dir
if defined NGRAM_NAME if not defined NGRAM_DIR exit /b 1

call :fetch_file "%MODEL_DIR%" "%ARTIFACT%" %EXPECTED_SIZE% %EXPECTED_SHA256% %REVISION% "%LABEL%" model
if not "!FETCH_OK!"=="1" exit /b 1
set "PART_N=1"
:fetch_parts
if %PART_N% gtr %PART_COUNT% goto :parts_done
call :fetch_file "%MODEL_DIR%" "%ARTIFACT%.part-000%PART_N%" !PART_%PART_N%_SIZE! !PART_%PART_N%_SHA256! %REVISION% "part %PART_N% of the model" part
if not "!FETCH_OK!"=="1" exit /b 1
set /a PART_N+=1
goto :fetch_parts
:parts_done
if defined NGRAM_NAME call :fetch_file "%NGRAM_DIR%" "%NGRAM_NAME%" %NGRAM_SIZE% %NGRAM_SHA256% %NGRAM_REVISION% "the n-gram volume (27.1 GiB)" volume
if defined NGRAM_NAME if not "!FETCH_OK!"=="1" exit /b 1

echo Model ready: !MODEL!
if defined NGRAM_NAME call :ngram_summary
echo Point the tests at it with:  set %TESTS_VARIABLE%=!MODEL!
call :fetch_graft
exit /b 0
:model_dir_rejected
echo The model folder "!MODEL_DIR!" has a percent sign, caret or exclamation mark in its name, which 1>&2
echo this script cannot pass on safely. Set NINFER_MODEL_DIR to a folder without those, or move this 1>&2
echo script, and run this file again. 1>&2
exit /b 1

rem Fetches one pinned file: folder %1, name %2, size %3, sha256 %4, revision %5, description %6, and
rem kind %7 (model, part or volume). Sets FETCH_OK=1 once the file is at its final name and verified.
rem A file already there that verifies is kept without any download.
rem
rem The download is staged under a revision-scoped name so that curl -C - can only ever resume the
rem same file. curl -C - resumes by appending at the current file length, without checking what wrote
rem those bytes, so a leftover partial from a different revision -- for instance the previous pin's,
rem on any machine that ran an older script -- would be spliced onto the head of this one: a file of
rem entirely plausible size that is corrupt throughout. A name that carries the revision means a
rem resume can only ever continue the same file, and the checks below are what promote it to the
rem final name. Written with goto rather than ( ) blocks: a folder may contain ")". Folders are read
rem with !VAR!, and any echo of one is given it quoted.
:fetch_file
set "FETCH_OK=0"
set "F_DIR=%~1"
set "F_NAME=%~2"
set "EXPECTED_SIZE=%~3"
set "EXPECTED_SHA256=%~4"
set "F_REV=%~5"
set "F_WHAT=%~6"
set "F_KIND=%~7"
set "F_PART_NAME=%F_NAME%.%F_REV%.part"
call :join_path F_PATH F_DIR "%F_NAME%"
call :join_path F_PART F_DIR "%F_PART_NAME%"
call :join_path F_DOT F_DIR "."
if not exist "!F_PATH!" goto :fetch_download
call :verify "!F_PATH!"
if not "!VERIFY_OK!"=="1" goto :fetch_stale
if /i "%F_KIND%"=="model" echo Model already present: !F_PATH!
if /i not "%F_KIND%"=="model" echo Already present: !F_PATH!
set "FETCH_OK=1"
exit /b 0
:fetch_stale
echo Existing !F_PATH! did not verify against revision %F_REV%; fetching the pinned one.
if /i not "%F_KIND%"=="model" goto :fetch_download
if defined NGRAM_NAME goto :fetch_download
echo If it is the v2 file from an earlier release, you can upgrade it instead of downloading:
echo   python tools\upgrade_ninfer_v2_to_v3.py "!F_PATH!" "!F_PATH!.v3" ^&^& move /y "!F_PATH!.v3" "!F_PATH!"
:fetch_download
echo Downloading %F_WHAT%...
set "URL=https://huggingface.co/%OWNER%/%REPO%/resolve/%F_REV%/%F_NAME%"
where /q aria2c
if errorlevel 1 goto :fetch_curl
rem 16 parallel ranges over one file, not single-stream curl. On the Vast rental boxes this repo
rem downloads at ~98 MB/s over aria2c against the same throttled-to-~1MB/s (and once outright
rem stalled at 11 of 21 GiB) single-stream curl -- see scripts\multi-gpu-testing\README.md. -c
rem resumes into the same .part name curl would use, and HuggingFace serves ranges fine.
rem The folder is given as "<dir>\." (F_DOT) because a drive root ends in a backslash, and a backslash
rem before the closing quote escapes the quote for aria2c's argument parser.
aria2c -x16 -s16 -c --file-allocation=none -d "!F_DOT!" -o "%F_PART_NAME%" "%URL%"
goto :fetch_downloaded
:fetch_curl
curl.exe -L -C - --fail --output "!F_PART!" "%URL%"
:fetch_downloaded
if errorlevel 1 goto :fetch_download_failed
call :verify "!F_PART!"
if not "!VERIFY_OK!"=="1" goto :fetch_unverified
move /y "!F_PART!" "!F_PATH!" >nul
if errorlevel 1 goto :fetch_move_failed
set "FETCH_OK=1"
exit /b 0
:fetch_download_failed
echo Download failed. Run this file again to resume.
exit /b 0
:fetch_unverified
echo Downloaded file at "!F_PART!" failed verification against revision %F_REV%. Delete it and run this file again.
exit /b 0
:fetch_move_failed
echo Failed to move "!F_PART!" to "!F_PATH!". Delete "!F_PART!" and run this file again.
exit /b 0

rem Sets NGRAM_DIR, the folder for the n-gram volume: NINFER_NGRAM_DIR, else the answer to a question,
rem else the model folder. Leaves NGRAM_DIR empty when the folder cannot be created or its name holds a
rem character the script cannot carry safely.
rem
rem Folder text is user input, so it is only ever read with !VAR! (delayed expansion), whose result cmd
rem does not parse again, and every command that shows or uses it is given it quoted or through !VAR!.
rem Splicing a percent expansion into an unquoted echo would run whatever follows an & or | in the folder name.
:choose_ngram_dir
set "NGRAM_DIR=!MODEL_DIR!"
if defined NINFER_NGRAM_DIR set "NGRAM_DIR=!NINFER_NGRAM_DIR!"
if defined NINFER_NGRAM_DIR set "NGRAM_DIR=!NGRAM_DIR:"=!"
if defined NINFER_NGRAM_DIR goto :ngram_dir_chosen
echo.
echo Qwen3.8-Flash-Next reads its 27.1 GiB n-gram volume on every prompt, and the model only once,
echo at load. Prompts were processed 1.1-1.4x faster with the volume on an NVMe drive than on a
echo SATA SSD, so put it on your fastest drive (28 GiB free). The model can stay where it is.
echo.
echo   Model folder: !MODEL_DIR_FULL!
echo.
rem set /p leaves the answer empty when nothing is typed or there is no console to read.
set "NGRAM_ANSWER="
set /p "NGRAM_ANSWER=Folder or drive for the n-gram volume, for example D: (Enter = the model folder): "
if not defined NGRAM_ANSWER goto :ngram_dir_chosen
set "NGRAM_ANSWER=!NGRAM_ANSWER:"=!"
if not defined NGRAM_ANSWER goto :ngram_dir_chosen
rem A bare drive gets a folder of its own rather than the drive's root. D: and D:\ are separate
rem branches, so the result of the first is never tested by the second (D: -> D:\ninfer-models).
if "!NGRAM_ANSWER:~1!"==":" goto :ngram_bare_drive
if "!NGRAM_ANSWER:~1!"==":\" goto :ngram_bare_drive_root
goto :ngram_answer_set
:ngram_bare_drive
set "NGRAM_ANSWER=!NGRAM_ANSWER!\ninfer-models"
goto :ngram_answer_set
:ngram_bare_drive_root
set "NGRAM_ANSWER=!NGRAM_ANSWER!ninfer-models"
:ngram_answer_set
set "NGRAM_DIR=!NGRAM_ANSWER!"
:ngram_dir_chosen
rem A value that was only quotes is empty after they are stripped: use the model folder.
if not defined NGRAM_DIR set "NGRAM_DIR=!MODEL_DIR!"
call :dir_check NGRAM_DIR
if "!DIR_BAD!"=="1" goto :ngram_dir_rejected
for %%I in ("%NGRAM_DIR%") do set "NGRAM_DIR=%%~fI"
call :trim_trailing_backslash NGRAM_DIR
call :join_path NGRAM_PROBE NGRAM_DIR "."
if not exist "!NGRAM_PROBE!" mkdir "%NGRAM_DIR%" >nul 2>&1
if exist "!NGRAM_PROBE!" goto :ngram_dir_ok
echo Cannot create the folder "!NGRAM_DIR!" for the n-gram volume. 1>&2
set "NGRAM_DIR="
exit /b 0
:ngram_dir_rejected
echo The n-gram folder "!NGRAM_DIR!" has a percent sign, caret or exclamation mark in its name, which 1>&2
echo this script cannot pass on safely. Choose a folder without those, and run this file again. 1>&2
set "NGRAM_DIR="
exit /b 0
:ngram_dir_ok
echo N-gram volume folder: !NGRAM_DIR!
exit /b 0

rem Sets DIR_BAD=1 when the variable named %1 (MODEL_DIR or NGRAM_DIR) holds a percent sign, caret or exclamation mark. The folder passes
rem through call, which doubles the first two, and through set and for lines, which eat the last, so it
rem would silently become a different folder. Ampersand, pipe, angle brackets and parentheses need no
rem check: the folder is quoted or read through delayed expansion everywhere, and never expanded inside
rem a ( ) block. MODEL_DIR_FULL is cleared before the MODEL_DIR check, so `set NAME` prints one variable.
rem
rem The folder is never expanded on a command line here. Earlier versions did, in substitutions such as
rem !NGRAM_DIR:^=!, where delayed expansion itself consumes the caret. Instead `set NGRAM_DIR` prints
rem "NGRAM_DIR=<folder>" and findstr looks for one of the three characters in that output. The findstr
rem line runs with delayed expansion off so its exclamation mark is an ordinary character; a doubled
rem percent sign is one percent sign in a batch file; the caret sits last in the class, where it is a
rem plain character. Each branch is a goto, so nothing conditional shares a line with an
rem unconditional command.
:dir_check
set "DIR_BAD="
setlocal disabledelayedexpansion
set %~1 2>nul | findstr /r "[!%%^]" >nul
if errorlevel 1 goto :dir_check_clean
endlocal
set "DIR_BAD=1"
exit /b 0
:dir_check_clean
endlocal
exit /b 0

rem Sets the variable named %1 to folder variable %2 joined with name %3, with exactly one backslash
rem between them: a drive root such as D:\ already ends in one. Folder text is read with !VAR! only.
:join_path
set "%~1=!%~2!"
if not "!%~2:~-1!"=="\" set "%~1=!%~1!\"
set "%~1=!%~1!%~3"
exit /b 0

rem Removes one trailing backslash from the variable named %1, except after a drive colon (D:\ is the
rem root, and D: alone would mean the current folder on that drive). for %%~fI keeps a trailing
rem backslash, which would double up in "<folder>\<name>" and fail a comparison with a folder
rem written without it.
:trim_trailing_backslash
if not "!%~1:~-1!"=="\" exit /b 0
if "!%~1:~-2,1!"==":" exit /b 0
set "%~1=!%~1:~0,-1!"
exit /b 0

rem The lines that start the server. run.bat looks in the same default models folder as this script
rem and prefers qwen4exp-iq2xs-mtp-vision.ninfer there, so it needs NINFER_MODEL_DIR only when that
rem was set here, and NINFER_NGRAM_VOLUME only when the volume is not beside the model (it reads
rem <model>.ngram otherwise). Not a ( ) block: a folder may contain ")".
:ngram_summary
call :join_path NGRAM_PATH NGRAM_DIR "%NGRAM_NAME%"
echo N-gram volume ready: !NGRAM_PATH!
echo.
echo Start it with:
if /i not "!NGRAM_DIR!"=="!MODEL_DIR_FULL!" echo   set "NINFER_NGRAM_VOLUME=!NGRAM_PATH!"
if defined NINFER_MODEL_DIR echo   set "NINFER_MODEL_DIR=!MODEL_DIR_FULL!"
echo   "!ROOT!run.bat" qwen38-flash-next
echo.
exit /b 0

rem The qwen38-27b godmode graft lives in a private repo. Fetching it is best-effort: with no token,
rem or a token without access, the model download still succeeds and the graft is simply skipped.
rem Both files are pinned to a revision and sha256 and land in artifacts\grafts beside the repo root,
rem where run.bat looks. NINFER_GRAFT_DIR overrides the directory, NINFER_GRAFTS=off skips the fetch.
:fetch_graft
if /i not "%ARTIFACT%"=="qwen3_8_27b.ninfer" exit /b 0
if /i "%NINFER_GRAFTS%"=="off" exit /b 0
set "G_REPO=WarlaxZ/Qwen3.8-27B-godmode-graft"
set "G_REV=97a02e9bbb37d76af63256152c07fc4eeb4c40c9"
set "G_DIR=!REPO_ROOT!\artifacts\grafts"
if defined NINFER_GRAFT_DIR set "G_DIR=!NINFER_GRAFT_DIR!"
set "G_BIN=godmode_q38_trained.bin"
set "G_BIN_SHA=a0ddecc4ca5db8b84f600d60fbf238c97e21a74069b7182e0e0e595d7a22c391"
set "G_JSON=godmode_q38_trained.json"
set "G_JSON_SHA=2a832ea95cc1dcee1dff2b292b10ef98913594eaf1f74121b01dd03a23dee3b1"
rem The graft is a pair run.bat trusts by the .bin alone and the loader needs whole with its .json
rem sidecar, so any failure below leaves neither file behind (:graft_abort).
set "G_COMPLETE=1"
call :graft_present %G_BIN% %G_BIN_SHA%
if not "!SHA_OK!"=="1" set "G_COMPLETE=0"
call :graft_present %G_JSON% %G_JSON_SHA%
if not "!SHA_OK!"=="1" set "G_COMPLETE=0"
if "!G_COMPLETE!"=="1" goto :graft_ready
rem The token is HF_TOKEN, else the file huggingface_hub reads: HF_TOKEN_PATH, or token under HF_HOME,
rem which itself defaults to huggingface\ under XDG_CACHE_HOME and then %USERPROFILE%\.cache.
set "G_TOKEN=%HF_TOKEN%"
set "G_TOKENFILE=%USERPROFILE%\.cache\huggingface\token"
if defined XDG_CACHE_HOME set "G_TOKENFILE=%XDG_CACHE_HOME%\huggingface\token"
if defined HF_HOME set "G_TOKENFILE=%HF_HOME%\token"
if defined HF_TOKEN_PATH set "G_TOKENFILE=%HF_TOKEN_PATH%"
if not defined G_TOKEN if exist "%G_TOKENFILE%" set /p G_TOKEN=<"%G_TOKENFILE%"
if not defined G_TOKEN goto :graft_no_token
if not exist "!G_DIR!" mkdir "!G_DIR!" >nul 2>&1
if not exist "!G_DIR!\" goto :graft_nodir
set "G_STOP="
call :graft_fetch %G_BIN% %G_BIN_SHA%
if defined G_STOP goto :graft_abort
call :graft_fetch %G_JSON% %G_JSON_SHA%
if defined G_STOP goto :graft_abort
rem Any pair already here is invalid or partial (a verified one went to :graft_ready). Clear it, binary first,
rem so the new pair never sits beside a leftover half of the old one. Then install the sidecar first and the
rem binary last: run.bat trusts the .bin alone, so it must never be visible without its .json.
del "!G_DIR!\%G_BIN%" "!G_DIR!\%G_JSON%" >nul 2>&1
move /y "!G_DIR!\%G_JSON%.part" "!G_DIR!\%G_JSON%" >nul 2>&1
if errorlevel 1 goto :graft_install_failed
move /y "!G_DIR!\%G_BIN%.part" "!G_DIR!\%G_BIN%" >nul 2>&1
if errorlevel 1 goto :graft_install_failed
:graft_ready
echo Graft ready: !G_DIR!\%G_BIN%
exit /b 0
:graft_no_token
rem Nothing can be fetched, but a lone half of the pair or a staged .part would still break a launch, so
rem drop those. A whole pair is left alone: it may be one the user supplied.
del "!G_DIR!\%G_BIN%.part" "!G_DIR!\%G_JSON%.part" >nul 2>&1
if exist "!G_DIR!\%G_BIN%" if not exist "!G_DIR!\%G_JSON%" del "!G_DIR!\%G_BIN%" >nul 2>&1
if exist "!G_DIR!\%G_JSON%" if not exist "!G_DIR!\%G_BIN%" del "!G_DIR!\%G_JSON%" >nul 2>&1
echo Graft: skipped, no Hugging Face token. Set HF_TOKEN or run "hf auth login" if you have access.
exit /b 0
:graft_nodir
echo Graft: skipped, cannot create !G_DIR!. 1>&2
exit /b 0
:graft_install_failed
echo Graft: skipped, cannot install into !G_DIR!. 1>&2
:graft_abort
del "!G_DIR!\%G_BIN%" "!G_DIR!\%G_JSON%" "!G_DIR!\%G_BIN%.part" "!G_DIR!\%G_JSON%.part" >nul 2>&1
exit /b 0

rem SHA_OK=1 when file %1 in the graft directory exists and its sha256 equals %2.
:graft_present
set "SHA_OK=0"
if not exist "!G_DIR!\%~1" exit /b 0
call :sha_ok "!G_DIR!\%~1" %~2
exit /b 0

rem Downloads one pinned graft file to its .part. Sets G_STOP on failure; the caller discards both files.
:graft_fetch
curl.exe -sSL --fail -H "Authorization: Bearer %G_TOKEN%" --output "!G_DIR!\%~1.part" "https://huggingface.co/%G_REPO%/resolve/%G_REV%/%~1" 2>nul
if errorlevel 1 goto :graft_denied
call :sha_ok "!G_DIR!\%~1.part" %~2
if not "!SHA_OK!"=="1" goto :graft_corrupt
exit /b 0
:graft_denied
echo Graft: skipped, no access to %G_REPO% with this token.
set "G_STOP=1"
exit /b 0
:graft_corrupt
echo Graft: %~1 failed sha256 verification, not installed. 1>&2
set "G_STOP=1"
exit /b 0

rem SHA_OK=1 when the sha256 of %1 equals %2.
:sha_ok
set "SHA_OK=0"
set "SHA_ACTUAL="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%~1" SHA256') do if not defined SHA_ACTUAL set "SHA_ACTUAL=%%H"
set "SHA_ACTUAL=!SHA_ACTUAL: =!"
if /i "!SHA_ACTUAL!"=="%~2" set "SHA_OK=1"
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
if "%NINFER_SKIP_SHA256%"=="1" goto :verify_skip_sha
set "ACTUAL_SHA256="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%VERIFY_PATH%" SHA256') do if not defined ACTUAL_SHA256 set "ACTUAL_SHA256=%%H"
set "ACTUAL_SHA256=!ACTUAL_SHA256: =!"
if /i "!ACTUAL_SHA256!"=="%EXPECTED_SHA256%" set "VERIFY_OK=1"
exit /b 0
:verify_skip_sha
set "VERIFY_OK=1"
exit /b 0
