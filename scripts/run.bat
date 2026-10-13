@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Serve a model on one RTX 3090.
rem
rem   run.bat [model] [profile]     (double-click it and it asks for the model)
rem
rem   model             profiles
rem   qwen38-27b        tuned (default), int8, c8   <- recommended
rem   qwen36-35b-a3b    tuned (default)
rem   qwen38-flash-next tuned (default)      <- Qwen3.8-Flash-Next from the IQ2_XS GGUF (preview)
rem
rem QWEN3.8-FLASH-NEXT: one request at a time, 32,768 tokens of context, INT8 KV, prefix reuse from
rem a 4 GiB Host cache (a conversation's next turn prefills only what is new; NINFER_PREFIX_CACHE=off
rem turns it off, NINFER_HOST_CONTEXT_MIB sizes it, NINFER_CONTEXT_NVME_MIB adds a disk tier,
rem NINFER_CONTEXT_STORE keeps it across restarts). MTP speculation (up to 3 drafts, 1.4x decode) is
rem on when the model is qwen4exp-iq2xs-mtp.ninfer or qwen4exp-iq2xs-mtp-vision.ninfer, preferred in
rem that order (vision first) when they sit beside the plain qwen4exp-iq2xs.ninfer; NINFER_SPEC=none
rem turns it off. Images are accepted with a -vision model (NINFER_VISION=off: text only). It pins all 33 GiB
rem of routed experts in host RAM and caches the most-used ones in VRAM; startup refuses, saying how
rem much is missing, when Windows cannot keep 2 GiB free beside them. The expert cache's state is
rem saved beside the model, so the second start is warm. NINFER_CONTEXT overrides the context.
rem The 29 GB n-gram volume (<model>.ngram) is read on every prompt, so keep it on the fastest drive:
rem with the model on a SATA SSD, a copy of only the volume on NVMe prefilled 1.1-1.4x faster.
rem Point NINFER_NGRAM_VOLUME at that copy (the 40 GB model can stay where it is):
rem
rem   set NINFER_NGRAM_VOLUME=C:\ninfer-models\qwen4exp\qwen4exp-iq2xs.ninfer.ngram
rem
rem One copy serves both the plain and the -mtp artifact when the -mtp one was converted with
rem --ngram-reuse, as docs\weight-conversion.md shows; the server refuses a volume from another
rem conversion.
rem
rem `tuned` is the recommended profile: rk4v4 KV, speculation plus the draft head, the memory
rem flags, vision in overlay residency, and an 8 GiB pinned Host context budget for the cache.
rem `int8` and `c8` are the older reference profiles for the 27B -- one user at 64K of INT8 KV
rem (the quality default), and eight lanes at 8K -- with every serving flag fixed.
rem
rem Every measurement behind these defaults, the memory model, and the reasoning for each flag are
rem in docs\maintainer\launcher-profiles.md. What follows is what you need to run it.
rem
rem QWEN3.8-27B, `tuned`: two flag sets, each measured (docs\performance.md, "Recommended
rem configurations"), chosen with NINFER_SPEC. The default is the fast one.
rem
rem   NINFER_SPEC=dflash2 (default): fastest at one stream, up to 188,416 tokens of context
rem
rem     --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15
rem     --prefill-cublas --prefill-chunk 4096
rem     --kv-dtype rk4v4 --gdn-state-fp16
rem     --vision --vision-residency overlay
rem
rem   NINFER_SPEC=mtp: the full 262,144-token native context, two lanes sharing it, still fast
rem
rem     --spec mtp --draft-tokens 3 --lm-head-draft
rem     --prefill-cublas --prefill-chunk 2048
rem     --kv-dtype rk4v4 --gdn-state-fp16
rem     --vision --vision-residency overlay
rem
rem   set NINFER_SPEC=mtp && run.bat qwen38-27b
rem
rem   MTP accepts NINFER_DRAFT_TOKENS up to 15. Three suits chat and prose; for coding work that
rem   returns edited files, 11-15 decodes up to 1.85x faster (docs\performance.md has the table).
rem
rem rk4v4 KV (Lloyd-Max 4-bit keys) is 31%% smaller than rk8v4 at the same decode speed, for +0.10%%
rem perplexity over it. Measured on a desktop RTX 3090 (2026-10-02, 1.6 GiB held by the desktop), the
rem DFlash2 set starts at 188,416 tokens with 721 MiB to spare and is refused at 196,608; its draft
rem weights are why it stops short of 262,144. The mtp set starts at 262,144 with two
rem lanes and about 1.2 GiB to spare. `none` is the mtp set without speculation.
rem The qwen3_8_27b.ninfer that download-model.bat fetches is the DFlash2 bundle and carries the
rem MTP weights too, so one file serves both. It stores the embedding as Q4 and the head as Q6, so
rem the --embedding-q4 and --lm-head-q6 load-time transcodes the upstream file needed are not passed.
rem
rem OVERRIDES, from the environment. All profiles: NINFER_MODEL (artifact path), NINFER_MODEL_DIR,
rem NINFER_SERVER, NINFER_HOST, NINFER_PORT, NINFER_GRAFT_DIR (phantom-kv graft directory),
rem NINFER_GRAFTS (set to "off" to disable graft loading), NINFER_DEFAULT_GRAFT (set to "on" to
rem apply the loaded graft to every request that names none), NINFER_CHAT_TEMPLATE (path to a local
rem Jinja file, passed straight to --chat-template; overrides the artifact's built-in template).
rem `tuned` also: NINFER_CONTEXT,
rem NINFER_CONCURRENCY, NINFER_KV_DTYPE, NINFER_SPEC, NINFER_DRAFT_TOKENS, NINFER_PREFILL_CHUNK,
rem NINFER_NGRAM (on^|off; n-gram copy drafting, on with DFlash2 at one or two lanes), NINFER_VISION (on^|off), NINFER_VISION_RESIDENCY, NINFER_HOST_CONTEXT_MIB (8192), NINFER_MIN_P (0.03) and
rem NINFER_PRESENCE_PENALTY (0.5), the loop guard ("default" keeps the registered preset).
rem
rem SERVING KNOBS, `tuned` only. Each is passed to ninfer-serve only when set, so leaving them all
rem unset keeps the profile exactly as measured; `ninfer-serve --help` says what each one does.
rem   NINFER_AUTO_HOST_CACHE=on           --auto-host-cache              size the Host context budget from free
rem                                                                      RAM instead of NINFER_HOST_CONTEXT_MIB
rem     NINFER_HOST_CACHE_PERCENT         --host-cache-percent N         share of free RAM to use (1-100)
rem     NINFER_HOST_CACHE_RESERVE_MIB     --host-cache-reserve-mib N     RAM to always leave free
rem     NINFER_HOST_CACHE_MAX_MIB         --host-cache-max-mib N         hard cap on pinned RAM
rem   NINFER_MAX_OUTPUT_TOKENS            --max-output-tokens N          cap every request's output budget
rem   NINFER_MLP_A8_DECODE=on             --mlp-a8-decode                INT8-activation MLP decode
rem   NINFER_REASONING_LOOP=stop^|conclude --reasoning-loop MODE         end (stop) or close the thinking
rem                                                                      (conclude) when it keeps repeating
rem                                                                      whole passages; off (the default)
rem                                                                      leaves thinking unguarded
rem   NINFER_CONTEXT_STORE               --context-store DIR            keep the context cache across restarts
rem                                                                      and crashes (dflash2, mtp, none)
rem     NINFER_CONTEXT_STORE_MAX_GIB      --context-store-max-gib N      disk budget for the store
rem
rem Each spec's defaults (context, lanes, chunk) are the ones measured to fit beside a desktop, which
rem holds roughly 1-1.5 GiB of the card.
rem
rem IF THE CARD IS BUSY. A desktop (or another job) holding VRAM can leave too little for the default
rem context. The `tuned` profiles' context is therefore a ceiling: the server is started with
rem --fit-context 32768 and, when the profile's context does not fit the GPU memory free at startup
rem with a 256 MiB safety margin, it starts at the largest context that does (a multiple of 1,024
rem tokens, at least 32,768) and prints one line saying so. If not even 32,768 tokens fit, it stops
rem and says how much memory is missing. Separately, if pinning the Host context budget fails (a
rem machine short of RAM), the launcher retries with half the budget, down to none. Values the
rem caller sets are honoured as given and fail loudly: an explicit NINFER_CONTEXT turns the fit off,
rem an explicit NINFER_HOST_CONTEXT_MIB (or NINFER_AUTO_HOST_CACHE=on) the host retry, and
rem NINFER_FALLBACK=off turns both off.
rem
rem Loopback by default. 0.0.0.0 publishes an unauthenticated OpenAI-compatible endpoint to every
rem network this machine is on, so it is opt-in per run rather than the shipped default:
rem
rem   set NINFER_HOST=0.0.0.0 && run.bat qwen38-27b
rem ---------------------------------------------------------------------------------------------

set "MODEL_KEY=%~1"
set "PROFILE=%~2"
rem Fit and host-retry eligibility, decided once on the first pass: only the tuned profile's
rem defaults may be second-guessed. A host-retry pass sets NINFER_HOST_CONTEXT_MIB itself and jumps
rem back to :model_known, so it must not count as the caller's choice.
if not defined NINFER_RUN_DECIDED (
  set "NINFER_RUN_DECIDED=1"
  set "HOST_RETRY=0"
  set "FIT=0"
  if "%NINFER_HOST_CONTEXT_MIB%"=="" if /i not "%NINFER_FALLBACK%"=="off" set "HOST_RETRY=1"
  if "%NINFER_CONTEXT%"=="" if /i not "%NINFER_FALLBACK%"=="off" set "FIT=1"
)
rem 32,768 tokens is the smallest context still worth serving; below it the server stops and says
rem how much memory is missing.
set "FIT_FLOOR=32768"
set "CONTEXT_WORD=context"
if "%FIT%"=="1" set "CONTEXT_WORD=context up to"
if "%PROFILE%"=="" set "PROFILE=tuned"
if "%MODEL_KEY%"=="" goto :choose_model
:model_resolve
if /i "%MODEL_KEY%"=="-h" goto :help
if /i "%MODEL_KEY%"=="--help" goto :help
if /i "%MODEL_KEY%"=="qwen38-27b" (
  set "ARTIFACT=qwen3_8_27b.ninfer"
  set "TITLE=Qwen3.8-27B"
  set "GRAFT_FILE=godmode_q38_trained.bin"
  goto :model_known
)
if /i "%MODEL_KEY%"=="qwen36-35b-a3b" (
  set "ARTIFACT=qwen3_6_35b_a3b.ninfer"
  set "TITLE=Qwen3.6-35B-A3B"
  goto :model_known
)
if /i "%MODEL_KEY%"=="qwen38-flash-next" (
  set "ARTIFACT=qwen4exp-iq2xs.ninfer"
  set "TITLE=Qwen3.8-Flash-Next (IQ2_XS)"
  goto :model_known
)
echo Unknown model: %MODEL_KEY% 1>&2
call :usage 1>&2
exit /b 2

:choose_model
rem Double-clicked from Explorer there is no argument to give, so ask. choice exits 255 when it has
rem no console to read from; that must not silently pick a model.
echo Which model?
echo   1  Qwen3.8-27B  (recommended)
echo   2  Qwen3.6-35B-A3B
echo   3  Qwen3.8-Flash-Next  (preview: one request at a time, 32K context)
choice /c 123 /n /m "Choose 1, 2 or 3: "
if errorlevel 255 exit /b 2
if errorlevel 3 (
  set "MODEL_KEY=qwen38-flash-next"
) else if errorlevel 2 (
  set "MODEL_KEY=qwen36-35b-a3b"
) else (
  set "MODEL_KEY=qwen38-27b"
)
goto :model_resolve

:help
call :usage
exit /b 0

:model_known
rem Two layouts share this launcher. In the release archive it sits at the archive root beside
rem models\, which is what download-model.bat writes to there. In a checkout it sits in scripts\,
rem one level under the repo root, and models\ (see .gitignore) is beside the repo root, not beside
rem this script -- a directory named scripts with a CMakeLists.txt above it tells the two apart (a
rem bare CMakeLists.txt probe would misfire on an archive unpacked beneath any source tree). This
rem must agree with download-model.bat's own default, which uses the same rule.
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
for %%I in ("%~dp0.") do set "SCRIPT_DIRNAME=%%~nxI"
set "MODEL_DIR=%~dp0models"
if /i "%SCRIPT_DIRNAME%"=="scripts" if exist "%ROOT%\CMakeLists.txt" set "MODEL_DIR=%ROOT%\models"
set "MODEL=%MODEL_DIR%\%ARTIFACT%"
rem An explicit NINFER_MODEL_DIR is taken verbatim and never probed, as in run.sh, so a model
rem downloaded there with download-model.bat is found here.
if not "%NINFER_MODEL_DIR%"=="" set "MODEL=%NINFER_MODEL_DIR%\%ARTIFACT%"
set "HOST=127.0.0.1"
set "PORT=8080"
set "KV_DTYPE=rk4v4"
if not "%NINFER_MODEL%"=="" set "MODEL=%NINFER_MODEL%"
if not "%NINFER_HOST%"=="" set "HOST=%NINFER_HOST%"
if not "%NINFER_PORT%"=="" set "PORT=%NINFER_PORT%"

set "SERVER=%ROOT%\build-ninja\apps\ninfer-serve.exe"
if not exist "%SERVER%" set "SERVER=%~dp0ninfer-serve.exe"
if not "%NINFER_SERVER%"=="" set "SERVER=%NINFER_SERVER%"

rem Phantom-KV graft: default directory is artifacts\grafts in this repo. The graft
rem file name is set per model key above. NINFER_GRAFTS=off disables graft loading entirely;
rem NINFER_GRAFT_DIR overrides where to look.
set "GRAFT_DIR=%ROOT%\artifacts\grafts"
if not "%NINFER_GRAFT_DIR%"=="" set "GRAFT_DIR=%NINFER_GRAFT_DIR%"

rem The profile fixes the whole serving shape. LABEL is the banner; PROFILE_ARGS is everything
rem after --host/--port. Values below use ^| for the separator: a bare pipe inside an expanded
rem variable would be parsed as a pipe when echoed.
set "LABEL="
set "PROFILE_ARGS="
set "PREFILL_NOTE="
set "HINT="
set "NGRAM_VOLUME_ARGS="
set "NGRAM_VOLUME_NOTE="
rem Only Flash-Next has an n-gram volume, so refuse rather than ignore the variable elsewhere.
rem Branch with goto rather than a parenthesised block, so the variable's value never sits inside one.
if "%NINFER_NGRAM_VOLUME%"=="" goto :ngram_guard_done
if /i "%MODEL_KEY%"=="qwen38-flash-next" goto :ngram_guard_done
echo NINFER_NGRAM_VOLUME is for qwen38-flash-next only; clear it to serve %MODEL_KEY% 1>&2
exit /b 2
:ngram_guard_done
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/tuned" goto :profile_27b_tuned
if /i "%MODEL_KEY%/%PROFILE%"=="qwen36-35b-a3b/tuned" goto :profile_35b_tuned
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-flash-next/tuned" goto :profile_flash_next
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/int8" goto :profile_27b_int8
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/c8" goto :profile_27b_c8
echo Model %MODEL_KEY% has no profile %PROFILE% 1>&2
call :usage 1>&2
exit /b 2

:profile_27b_tuned
rem The speculative backend fixes everything that has to move with it: the prefill chunk (the
rem cuBLAS route's workspace scales with it) and the context that fits. The overrides are applied after these defaults, so an explicit
rem NINFER_CONTEXT or NINFER_PREFILL_CHUNK always wins.
set "SPEC=dflash2"
if not "%NINFER_SPEC%"=="" set "SPEC=%NINFER_SPEC%"
if /i "%SPEC%"=="dflash2" goto :spec_dflash2
if /i "%SPEC%"=="mtp" goto :spec_mtp
if /i "%SPEC%"=="none" goto :spec_none
echo NINFER_SPEC must be dflash2, mtp or none, got %SPEC% 1>&2
exit /b 2

:spec_dflash2
set "SPEC=dflash2"
set "CONTEXT=188416"
set "CONCURRENCY=1"
set "PREFILL_CHUNK=4096"
set "DRAFT_TOKENS=7"
goto :spec_done

:spec_mtp
set "SPEC=mtp"
set "CONTEXT=262144"
set "CONCURRENCY=2"
set "PREFILL_CHUNK=2048"
set "DRAFT_TOKENS=3"
goto :spec_done

:spec_none
set "SPEC=none"
set "CONTEXT=262144"
set "CONCURRENCY=2"
set "PREFILL_CHUNK=2048"
set "DRAFT_TOKENS="

:spec_done
if not "%NINFER_DRAFT_TOKENS%"=="" set "DRAFT_TOKENS=%NINFER_DRAFT_TOKENS%"
if not "%NINFER_CONTEXT%"=="" set "CONTEXT=%NINFER_CONTEXT%"
if not "%NINFER_CONCURRENCY%"=="" set "CONCURRENCY=%NINFER_CONCURRENCY%"
if not "%NINFER_KV_DTYPE%"=="" set "KV_DTYPE=%NINFER_KV_DTYPE%"
if not "%NINFER_PREFILL_CHUNK%"=="" set "PREFILL_CHUNK=%NINFER_PREFILL_CHUNK%"
rem N-gram copy drafting rides on DFlash2 and is on at one and two lanes: +42%% decode at one lane on
rem agent traffic that writes back files it has read, +15%% batch throughput at two, where a lane
rem without a copy is widened only as far as its own balance allows (about 3%% of its round time).
rem Three or more lanes never copy, so it stays off there unless NINFER_NGRAM=on.
set "NGRAM=off"
if /i "%SPEC%"=="dflash2" if "%CONCURRENCY%"=="1" set "NGRAM=on"
if /i "%SPEC%"=="dflash2" if "%CONCURRENCY%"=="2" set "NGRAM=on"
if not "%NINFER_NGRAM%"=="" set "NGRAM=%NINFER_NGRAM%"
if /i not "%NGRAM%"=="on" if /i not "%NGRAM%"=="off" (
  echo NINFER_NGRAM must be on or off, got %NGRAM% 1>&2
  exit /b 2
)
if /i "%NGRAM%"=="on" if /i not "%SPEC%"=="dflash2" (
  echo NINFER_NGRAM=on needs NINFER_SPEC=dflash2 1>&2
  exit /b 2
)
set "SPEC_ARGS="
if /i not "%SPEC%"=="none" set "SPEC_ARGS=--spec %SPEC% --draft-tokens %DRAFT_TOKENS% --lm-head-draft"
if /i "%NGRAM%"=="on" set "SPEC_ARGS=%SPEC_ARGS% --ngram-draft-tokens 15"
if /i "%SPEC%"=="dflash2" set "SPEC_LABEL=DFlash2 K=%DRAFT_TOKENS% + draft head"
if /i "%NGRAM%"=="on" set "SPEC_LABEL=%SPEC_LABEL% + n-gram copies"
if /i "%SPEC%"=="mtp" set "SPEC_LABEL=MTP%DRAFT_TOKENS% + draft head, full context"
if /i "%SPEC%"=="none" set "SPEC_LABEL=no speculation"
set "PROFILE_ARGS=--max-concurrency %CONCURRENCY% --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-dtype %KV_DTYPE% %SPEC_ARGS% --gdn-state-fp16 --prefill-cublas --prefill-chunk %PREFILL_CHUNK%"
set "LABEL=C%CONCURRENCY%  ^|  %CONTEXT_WORD% %CONTEXT%  ^|  %KV_DTYPE% KV  ^|  %SPEC_LABEL%"
set "PREFILL_NOTE=Prefill: cuBLAS route, chunk %PREFILL_CHUNK%"
if /i "%SPEC%"=="dflash2" set "HINT=Need the full 262K context or a second lane? Set NINFER_SPEC=mtp: slower decode."
goto :profile_tuned_common

:profile_35b_tuned
rem rk4v4 KV fits the full native context with two lanes beside a desktop on the default prefill
rem route (three measured to start, 2026-09-24; rk8v4 managed 147,456 with one); the cuBLAS route
rem below trades some of it for prefill speed. DFlash2 is a 27B-only backend. Lanes share the
rem one --kv-capacity pool: any request may use all 262,144 tokens, but the lanes' requests together
rem hold at most that many at a time.
set "SPEC=mtp"
if not "%NINFER_SPEC%"=="" set "SPEC=%NINFER_SPEC%"
rem cuBLAS prefill at chunk 4096: 8,848 tok/s on a 4K prompt against 5,470 at the old default-route
rem chunk 512 (2026-10-02). Its larger runtime reservation is what costs context: beside a desktop
rem holding 1.3 GiB the old profile started at 262,144 with 154 MiB free, while this one is refused
rem at 229,376 (27 MB short) and starts at 212,992 with 220 MiB free, so that is the default. For
rem the full 262,144, set NINFER_PREFILL_CHUNK=1024 (7,140 tok/s, fits with 138 MiB free).
set "CONTEXT=212992"
set "CONCURRENCY=2"
set "PREFILL_CHUNK=4096"
set "DRAFT_TOKENS=3"
rem DFlash2 is 27B-only and n-gram copy rounds ride on it, so refuse rather than ignore the flag.
if "%NINFER_NGRAM%"=="" goto :ngram35_done
if /i "%NINFER_NGRAM%"=="off" goto :ngram35_done
if /i "%NINFER_NGRAM%"=="on" (
  echo NINFER_NGRAM=on needs NINFER_SPEC=dflash2, which only qwen38-27b has 1>&2
  exit /b 2
)
echo NINFER_NGRAM must be on or off, got %NINFER_NGRAM% 1>&2
exit /b 2
:ngram35_done
if /i "%SPEC%"=="mtp" goto :spec35_mtp
if /i "%SPEC%"=="none" goto :spec35_none
echo NINFER_SPEC must be mtp or none, got %SPEC% 1>&2
exit /b 2
:spec35_mtp
set "SPEC=mtp"
set "SPEC_LABEL=MTP3 + draft head"
goto :spec35_done
:spec35_none
set "SPEC=none"
set "SPEC_LABEL=no speculation"
:spec35_done
if not "%NINFER_DRAFT_TOKENS%"=="" set "DRAFT_TOKENS=%NINFER_DRAFT_TOKENS%"
if not "%NINFER_CONTEXT%"=="" set "CONTEXT=%NINFER_CONTEXT%"
if not "%NINFER_CONCURRENCY%"=="" set "CONCURRENCY=%NINFER_CONCURRENCY%"
if not "%NINFER_KV_DTYPE%"=="" set "KV_DTYPE=%NINFER_KV_DTYPE%"
if not "%NINFER_PREFILL_CHUNK%"=="" set "PREFILL_CHUNK=%NINFER_PREFILL_CHUNK%"
set "SPEC_ARGS="
if /i "%SPEC%"=="mtp" set "SPEC_ARGS=--spec mtp --draft-tokens %DRAFT_TOKENS% --lm-head-draft --mtp-experts-q4"
if /i "%SPEC%"=="mtp" set "SPEC_LABEL=MTP%DRAFT_TOKENS% + draft head"
set "PROFILE_ARGS=--max-concurrency %CONCURRENCY% --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-dtype %KV_DTYPE% %SPEC_ARGS% --gdn-state-fp16 --prefill-cublas --prefill-chunk %PREFILL_CHUNK%"
set "LABEL=C%CONCURRENCY%  ^|  %CONTEXT_WORD% %CONTEXT%  ^|  %KV_DTYPE% KV  ^|  %SPEC_LABEL%"
goto :profile_tuned_common

:profile_flash_next
rem The only serving shape this release supports: one lane, INT8 KV. The step-down ladder does not
rem apply (its refusals are another model's); a refusal here says what to change. PROFILE is
rem renamed so the tuned profiles' cache and sampling notes are not printed for it.
set "LADDER=0"
set "PROFILE=flash-next"
set "CONTEXT=32768"
if not "%NINFER_CONTEXT%"=="" set "CONTEXT=%NINFER_CONTEXT%"
rem MTP speculation needs the artifact converted with the drafter (--mtp-gguf). Without an explicit
rem NINFER_MODEL, that artifact is preferred when it sits beside the plain one; speculation is on
rem when the model is it. NINFER_SPEC=none turns it off, NINFER_SPEC=mtp forces it (a plain artifact
rem then refuses at startup with the conversion to run). A ceiling of 3 drafts measured best or tied
rem at 1K-32K (docs\maintainer\flash-next-plan.md, PR 13); each round drafts what acceptance pays for.
rem Image input needs the artifact converted with the vision tower (--mmproj); the one with the
rem drafter and the tower (qwen4exp-iq2xs-mtp-vision.ninfer) is preferred over the drafter-only one,
rem and --vision is passed when the model is a -vision artifact (NINFER_VISION=off drops it).
for %%F in ("%MODEL%") do set "MODEL_FOLDER=%%~dpF"
if "%NINFER_MODEL%"=="" if exist "%MODEL_FOLDER%qwen4exp-iq2xs-mtp.ninfer" set "MODEL=%MODEL_FOLDER%qwen4exp-iq2xs-mtp.ninfer"
if "%NINFER_MODEL%"=="" if exist "%MODEL_FOLDER%qwen4exp-iq2xs-mtp-vision.ninfer" set "MODEL=%MODEL_FOLDER%qwen4exp-iq2xs-mtp-vision.ninfer"
set "SPEC=none"
set "FN_VISION=off"
for %%F in ("%MODEL%") do if /i "%%~nF"=="qwen4exp-iq2xs-mtp" set "SPEC=mtp"
for %%F in ("%MODEL%") do if /i "%%~nF"=="qwen4exp-iq2xs-mtp-vision" set "SPEC=mtp"
for %%F in ("%MODEL%") do if /i "%%~nF"=="qwen4exp-iq2xs-mtp-vision" set "FN_VISION=on"
for %%F in ("%MODEL%") do if /i "%%~nF"=="qwen4exp-iq2xs-vision" set "FN_VISION=on"
if not "%NINFER_SPEC%"=="" set "SPEC=%NINFER_SPEC%"
if not "%NINFER_VISION%"=="" set "FN_VISION=%NINFER_VISION%"
if /i not "%FN_VISION%"=="on" if /i not "%FN_VISION%"=="off" (
  echo NINFER_VISION must be on or off, got %FN_VISION% 1>&2
  exit /b 2
)
set "FN_VISION_ARGS="
set "FN_INPUT=text only"
if /i "%FN_VISION%"=="on" set "FN_VISION_ARGS=--vision"
if /i "%FN_VISION%"=="on" set "FN_INPUT=text and images"
set "DRAFT_TOKENS=3"
if not "%NINFER_DRAFT_TOKENS%"=="" set "DRAFT_TOKENS=%NINFER_DRAFT_TOKENS%"
set "SPEC_ARGS="
set "SPEC_LABEL=no speculation"
if /i "%SPEC%"=="mtp" set "SPEC_ARGS=--spec mtp --draft-tokens %DRAFT_TOKENS%"
if /i "%SPEC%"=="mtp" set "SPEC_LABEL=MTP, up to %DRAFT_TOKENS% drafts"
if /i not "%SPEC%"=="mtp" if /i not "%SPEC%"=="none" (
  echo NINFER_SPEC must be mtp or none for Qwen3.8-Flash-Next, got %SPEC% 1>&2
  exit /b 2
)
rem Prefix reuse (the context cache): checkpoints of the conversation in a pinned Host tier beside the
rem experts, 4 GiB by default; the server shrinks it to what its RAM ledger can spare (off below
rem 512 MiB) and says so. NINFER_PREFIX_CACHE=off turns it off; NINFER_HOST_CONTEXT_MIB sets its size.
set "FN_CACHE=on"
if not "%NINFER_PREFIX_CACHE%"=="" set "FN_CACHE=%NINFER_PREFIX_CACHE%"
if /i not "%FN_CACHE%"=="on" if /i not "%FN_CACHE%"=="off" (
  echo NINFER_PREFIX_CACHE must be on or off, got %FN_CACHE% 1>&2
  exit /b 2
)
set "FN_CACHE_ARGS="
set "FN_CACHE_LABEL=prefix cache"
if not "%NINFER_HOST_CONTEXT_MIB%"=="" set "FN_CACHE_ARGS=--host-context-mib %NINFER_HOST_CONTEXT_MIB%"
rem NINFER_CONTEXT_NVME_MIB=N also keeps checkpoints in a scratch file of up to N MiB beside the model
rem (deleted at exit), the host cache its RAM cache; with NINFER_HOST_CONTEXT_MIB=0, disk only.
if not "%NINFER_CONTEXT_NVME_MIB%"=="" set "FN_CACHE_ARGS=%FN_CACHE_ARGS% --context-nvme-mib %NINFER_CONTEXT_NVME_MIB%"
if not "%NINFER_CONTEXT_NVME_MIB%"=="" set "FN_CACHE_LABEL=prefix cache + disk"
if /i "%FN_CACHE%"=="off" set "FN_CACHE_ARGS=--host-context-mib 0"
if /i "%FN_CACHE%"=="off" set "FN_CACHE_LABEL=no prefix cache"
rem NINFER_CONTEXT_STORE=DIR keeps the conversations across restarts (--context-store); not with the
rem cache off. The unquoted set form keeps metacharacters in the path inert (as below).
set "STORE_ARGS="
if /i "%FN_CACHE%"=="on" if not "%NINFER_CONTEXT_STORE%"=="" set STORE_ARGS=--context-store "%NINFER_CONTEXT_STORE%"
if /i "%FN_CACHE%"=="on" if not "%NINFER_CONTEXT_STORE%"=="" if not "%NINFER_CONTEXT_STORE_MAX_GIB%"=="" set STORE_ARGS=%STORE_ARGS% --context-store-max-gib %NINFER_CONTEXT_STORE_MAX_GIB%
set "PROFILE_ARGS=--max-concurrency 1 --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-dtype int8 --prefill-chunk 1024 --max-pending-requests 16 --pending-timeout-ms 600000 %SPEC_ARGS% %FN_VISION_ARGS% %FN_CACHE_ARGS%"
set "LABEL=one request  ^|  context %CONTEXT%  ^|  INT8 KV  ^|  %FN_INPUT%, %SPEC_LABEL%  ^|  %FN_CACHE_LABEL%"
if /i "%FN_VISION%"=="off" if "%NINFER_VISION%"=="" set "HINT=Images: convert with --mtp-gguf and --mmproj to qwen4exp-iq2xs-mtp-vision.ninfer (docs\weight-conversion.md); this launcher then uses it."
if /i "%FN_VISION%"=="off" if "%NINFER_VISION%"=="" if not "%NINFER_MODEL%"=="" if exist "%MODEL_FOLDER%qwen4exp-iq2xs-mtp-vision.ninfer" set "HINT=Images: qwen4exp-iq2xs-mtp-vision.ninfer is beside this model; clear NINFER_MODEL to use it."
if /i "%SPEC%"=="none" if exist "%MODEL%" for %%F in ("%MODEL%") do if /i not "%%~nF"=="qwen4exp-iq2xs-mtp" if /i not "%%~nF"=="qwen4exp-iq2xs-mtp-vision" set "HINT=Faster decode: convert with --mtp-gguf to qwen4exp-iq2xs-mtp.ninfer (docs\weight-conversion.md); this launcher then uses it."
if /i "%SPEC%"=="none" if not "%NINFER_MODEL%"=="" if exist "%MODEL_FOLDER%qwen4exp-iq2xs-mtp.ninfer" for %%F in ("%MODEL%") do if /i not "%%~nF"=="qwen4exp-iq2xs-mtp" if /i not "%%~nF"=="qwen4exp-iq2xs-mtp-vision" set "HINT=Faster decode: qwen4exp-iq2xs-mtp.ninfer is beside this model; clear NINFER_MODEL to use it."
set "PREFILL_NOTE=Loads in about 2 minutes: 33 GiB of experts are pinned in host RAM"
rem The 29 GB n-gram volume is read unbuffered on every prompt, so its drive sets prefill speed while
rem the 40 GB artifact is read once at load. Measured 2026-10-12 with the artifact on a SATA SSD:
rem moving only the volume to NVMe made prefill 1.37x / 1.11x / 1.13x faster at 4K / 32K / 128K
rem (mean per-prompt ratio; docs\serving.md). NINFER_NGRAM_VOLUME points at such a copy; the server
rem checks it belongs to the model, so one copy serves the plain and the -mtp artifact alike when they
rem share a volume (--ngram-reuse). Not a parenthesised block: the path may contain ")".
set "NGRAM_VOLUME_NOTE=N-gram volume: beside the model. It is read on every prompt; if that is a SATA drive, a copy on NVMe (NINFER_NGRAM_VOLUME) prefills 1.1-1.4x faster"
if "%NINFER_NGRAM_VOLUME%"=="" goto :ngram_volume_done
if exist "%NINFER_NGRAM_VOLUME%\*" goto :ngram_volume_folder
if not exist "%NINFER_NGRAM_VOLUME%" goto :ngram_volume_missing
set NGRAM_VOLUME_ARGS=--ngram-volume "%NINFER_NGRAM_VOLUME%"
set NGRAM_VOLUME_NOTE=N-gram volume: "%NINFER_NGRAM_VOLUME%"  (NINFER_NGRAM_VOLUME)
goto :ngram_volume_done
:ngram_volume_folder
echo NINFER_NGRAM_VOLUME must name the .ngram file itself, not a folder: "%NINFER_NGRAM_VOLUME%" 1>&2
exit /b 2
:ngram_volume_missing
echo NINFER_NGRAM_VOLUME names a file that does not exist: "%NINFER_NGRAM_VOLUME%" 1>&2
echo Copy the model's .ngram file there, or clear NINFER_NGRAM_VOLUME to read the one beside the model. 1>&2
exit /b 2
:ngram_volume_done
goto :launch

:profile_27b_int8
set "PROFILE_ARGS=--max-context 65536 --kv-capacity 65536 --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 --prefill-chunk 1024 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft"
set "LABEL=one request  ^|  64K context  ^|  INT8 KV  ^|  MTP3, ReplaySSM"
goto :launch

:profile_27b_c8
rem Context cache sized per lane, so several agents rotating through the lanes find their own
rem conversation still cached instead of re-prefilling it: one checkpoint StateImage per lane on
rem the card beyond the active ones, and the same 8 GiB pinned Host context budget as `tuned` for
rem the conversations that leave the card. MEMORY COST: this profile keeps the GDN state in BF16, so
rem a StateImage is 147 MiB. The device slots take 8 x 147 MiB = 1.15 GiB of VRAM, the engine default
rem at eight lanes; the Host budget pins 8 GiB of RAM (StateImages, KV pages and pause snapshots
rem share it). Retention on the card is still bounded by the 16,384-token KV pool below.
set "C8_LANES=8"
set /a C8_DEVICE_STATES=C8_LANES
set "PROFILE_ARGS=--max-context 8192 --kv-capacity 16384 --max-concurrency %C8_LANES% --max-pending-requests 32 --pending-timeout-ms 600000 --prefill-chunk 512 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --device-state-slots %C8_DEVICE_STATES% --host-context-mib 8192"
set "LABEL=up to eight requests  ^|  8K context  ^|  INT8 KV  ^|  MTP3, ReplaySSM"
goto :launch

:profile_tuned_common
rem Only `tuned` carries vision and the sampling guard: the reference profiles are deliberately
rem minimal. Vision stays on -- overlay residency keeps the tower host-pinned and streams each
rem image through a borrowed device window, so it costs about 10 MiB of runtime reservation.
set "VISION=on"
if not "%NINFER_VISION%"=="" set "VISION=%NINFER_VISION%"
set "VISION_RESIDENCY=overlay"
if not "%NINFER_VISION_RESIDENCY%"=="" set "VISION_RESIDENCY=%NINFER_VISION_RESIDENCY%"
set "VISION_ARGS="
rem Pinned Host context budget, in MiB: one byte budget that retained StateImages (74.5 MiB each on
rem the 27B with the FP16 state), KV pages and pause snapshots share once they leave the card. It is
rem host RAM, pinned in full at startup on Windows as on Linux -- current drivers no longer charge
rem pinned memory against free VRAM, so it costs no context. Lower it if the box is short on RAM;
rem 0 keeps everything on the card.
set "HOST_CONTEXT_MIB=8192"
if not "%NINFER_HOST_CONTEXT_MIB%"=="" set "HOST_CONTEXT_MIB=%NINFER_HOST_CONTEXT_MIB%"
if /i "%VISION%"=="on" (
  set "VISION_ARGS=--vision --vision-residency %VISION_RESIDENCY%"
  set "LABEL=%LABEL%  ^|  vision (%VISION_RESIDENCY%)"
  goto :vision_done
)
if /i "%VISION%"=="off" (
  set "LABEL=%LABEL%  ^|  text only"
  goto :vision_done
)
echo NINFER_VISION must be on or off, got %VISION% 1>&2
exit /b 2
:vision_done
rem Loop guard for the small quant: a mild min-p trims the noisy token tail, and a mild presence
rem penalty breaks repetition loops in the reasoning and the answer. These are process-level
rem overrides, so they replace the registered presets in both thinking and non-thinking mode
rem (thinking presence 0, non-thinking 1.5); a request that sets its own value still wins. Kept
rem low because code legitimately repeats identifiers and syntax -- a penalty near 1.5 or any
rem frequency penalty damages generated code. Temperature, top-p and top-k stay at the registered
rem values. "default" omits the flag and leaves the registered preset in force.
set "MIN_P=0.03"
if not "%NINFER_MIN_P%"=="" set "MIN_P=%NINFER_MIN_P%"
set "PRESENCE=0.5"
if not "%NINFER_PRESENCE_PENALTY%"=="" set "PRESENCE=%NINFER_PRESENCE_PENALTY%"
set "SAMPLING_ARGS="
if /i not "%MIN_P%"=="default" set "SAMPLING_ARGS=--min-p %MIN_P%"
if /i not "%PRESENCE%"=="default" set "SAMPLING_ARGS=%SAMPLING_ARGS% --presence-penalty %PRESENCE%"
rem --auto-host-cache sizes that budget from the RAM still free once the model has loaded, and
rem refuses --host-context-mib, so it is either/or: with it on, NINFER_HOST_CONTEXT_MIB (and the
rem host retry's smaller budget) is ignored.
set "AUTO_HOST_CACHE=off"
if not "%NINFER_AUTO_HOST_CACHE%"=="" set "AUTO_HOST_CACHE=%NINFER_AUTO_HOST_CACHE%"
set "CACHE_ARGS="
if /i "%AUTO_HOST_CACHE%"=="on" goto :cache_auto
if /i "%AUTO_HOST_CACHE%"=="off" goto :cache_fixed
echo NINFER_AUTO_HOST_CACHE must be on or off, got %AUTO_HOST_CACHE% 1>&2
exit /b 2
:cache_auto
set "CACHE_ARGS=--auto-host-cache"
set "HOST_RETRY=0"
if not "%NINFER_HOST_CACHE_PERCENT%"=="" set "CACHE_ARGS=%CACHE_ARGS% --host-cache-percent %NINFER_HOST_CACHE_PERCENT%"
if not "%NINFER_HOST_CACHE_RESERVE_MIB%"=="" set "CACHE_ARGS=%CACHE_ARGS% --host-cache-reserve-mib %NINFER_HOST_CACHE_RESERVE_MIB%"
if not "%NINFER_HOST_CACHE_MAX_MIB%"=="" set "CACHE_ARGS=%CACHE_ARGS% --host-cache-max-mib %NINFER_HOST_CACHE_MAX_MIB%"
set "CACHE_NOTE=Context cache: sized automatically from free host RAM  (NINFER_AUTO_HOST_CACHE)"
goto :cache_done
:cache_fixed
if not "%NINFER_HOST_CACHE_PERCENT%%NINFER_HOST_CACHE_RESERVE_MIB%%NINFER_HOST_CACHE_MAX_MIB%"=="" (
  echo NINFER_HOST_CACHE_* needs NINFER_AUTO_HOST_CACHE=on 1>&2
  exit /b 2
)
set "CACHE_ARGS=--host-context-mib %HOST_CONTEXT_MIB%"
set "CACHE_NOTE=Context cache: %HOST_CONTEXT_MIB% MiB pinned host RAM  (NINFER_HOST_CONTEXT_MIB)"
:cache_done
rem Opt-in serving knobs: each is appended only when set, so the defaults above are untouched.
set "KNOB_ARGS="
if not "%NINFER_MAX_OUTPUT_TOKENS%"=="" set "KNOB_ARGS=%KNOB_ARGS% --max-output-tokens %NINFER_MAX_OUTPUT_TOKENS%"
if /i "%NINFER_MLP_A8_DECODE%"=="on" set "KNOB_ARGS=%KNOB_ARGS% --mlp-a8-decode"
if not "%NINFER_MLP_A8_DECODE%"=="" if /i not "%NINFER_MLP_A8_DECODE%"=="on" if /i not "%NINFER_MLP_A8_DECODE%"=="off" (
  echo NINFER_MLP_A8_DECODE must be on or off, got %NINFER_MLP_A8_DECODE% 1>&2
  exit /b 2
)
rem Reasoning-loop guard (--reasoning-loop): off unless asked for. It has never fired on the 27B in
rem the measurements behind it (#217: 21 requests, including one written to make the model loop),
rem and its thresholds come from other models, so turning it on by default would trade an unmeasured
rem false-positive risk for no observed benefit. `off` is the server's own default and passes nothing.
set "REASONING_LOOP="
set "LOOP_NOTE="
if "%NINFER_REASONING_LOOP%"=="" goto :loop_done
if /i "%NINFER_REASONING_LOOP%"=="off" (
  set "LOOP_NOTE=Reasoning-loop guard: off  (NINFER_REASONING_LOOP)"
  goto :loop_done
)
if /i "%NINFER_REASONING_LOOP%"=="stop" (
  set "REASONING_LOOP=stop"
  set "LOOP_NOTE=Reasoning-loop guard: stop - a reply whose thinking keeps repeating itself ends early  (NINFER_REASONING_LOOP)"
  goto :loop_done
)
if /i "%NINFER_REASONING_LOOP%"=="conclude" (
  set "REASONING_LOOP=conclude"
  set "LOOP_NOTE=Reasoning-loop guard: conclude - thinking that keeps repeating itself is closed and the model answers  (NINFER_REASONING_LOOP)"
  goto :loop_done
)
echo NINFER_REASONING_LOOP must be off, stop or conclude, got %NINFER_REASONING_LOOP% 1>&2
exit /b 2
:loop_done
if defined REASONING_LOOP set "KNOB_ARGS=%KNOB_ARGS% --reasoning-loop %REASONING_LOOP%"
rem The store directory is an arbitrary path: the unquoted `set NAME=value` form puts the quotes
rem around the expansion, as for CHAT_TEMPLATE_ARGS below, so metacharacters stay inert.
set "STORE_ARGS="
if not "%NINFER_CONTEXT_STORE%"=="" set STORE_ARGS=--context-store "%NINFER_CONTEXT_STORE%"
if not "%NINFER_CONTEXT_STORE%"=="" if not "%NINFER_CONTEXT_STORE_MAX_GIB%"=="" set STORE_ARGS=%STORE_ARGS% --context-store-max-gib %NINFER_CONTEXT_STORE_MAX_GIB%
if "%NINFER_CONTEXT_STORE%"=="" if not "%NINFER_CONTEXT_STORE_MAX_GIB%"=="" (
  echo NINFER_CONTEXT_STORE_MAX_GIB needs NINFER_CONTEXT_STORE 1>&2
  exit /b 2
)
rem The profile's context is a ceiling unless the caller chose it: --fit-context lets the server start
rem at the largest context that fits the GPU memory free right now (never below FIT_FLOOR) and say so.
set "FIT_ARGS="
if "%FIT%"=="1" set "FIT_ARGS= --fit-context %FIT_FLOOR%"
set "PROFILE_ARGS=%PROFILE_ARGS% --max-pending-requests 16 --pending-timeout-ms 600000 %VISION_ARGS% %CACHE_ARGS%%KNOB_ARGS%%FIT_ARGS% %SAMPLING_ARGS%"

:launch
set "GRAFT_ARGS="
if /i "%NINFER_GRAFTS%"=="off" goto :graft_done
if "%GRAFT_FILE%"=="" goto :graft_done
if exist "%GRAFT_DIR%\%GRAFT_FILE%" goto :graft_found
rem No parenthesised block here: GRAFT_DIR may contain ")" (e.g. "Program Files (x86)").
echo WARNING: graft file not found, serving without a graft: "%GRAFT_DIR%\%GRAFT_FILE%"
echo          Requests naming "godmode" will fail with unknown_graft.
goto :graft_done
:graft_found
set "GRAFT_ARGS=--graft "godmode=%GRAFT_DIR%\%GRAFT_FILE%""
rem Opt-in: NINFER_DEFAULT_GRAFT=on makes godmode apply to every request that states no graft
rem (--default-graft godmode); a request opts out with "graft": "". Only reached when a graft loaded.
if /i "%NINFER_DEFAULT_GRAFT%"=="on" set "GRAFT_ARGS=%GRAFT_ARGS% --default-graft godmode"
:graft_done
set "CHAT_TEMPLATE_ARGS="
rem The wrapping "set "VAR=..."" form closes its quoted span right before a spliced %VAR%, so cmd
rem would parse the expanded text unquoted and a value containing "&", "|", "<" or ">" could break
rem out of the statement and run as a separate command (delayed expansion would dodge that but
rem corrupts any "!" in every other %VAR% read later in the script -- see the prior round). The
rem unquoted `set NAME=value` form below avoids both: it opens its own quote immediately before
rem %NINFER_CHAT_TEMPLATE% and closes it right after, so the expansion lands inside a quoted span
rem (metacharacters inert) while still storing that literal pair of quotes around the path, same
rem as GRAFT_ARGS above.
if not "%NINFER_CHAT_TEMPLATE%"=="" set CHAT_TEMPLATE_ARGS=--chat-template "%NINFER_CHAT_TEMPLATE%"
if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Build it first:  .\scripts\build.ps1
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Download it first:  download-model.bat %MODEL_KEY%
  exit /b 1
)

echo %TITLE%  ^|  %LABEL%
if not "%PREFILL_NOTE%"=="" echo %PREFILL_NOTE%
if defined NGRAM_VOLUME_NOTE echo %NGRAM_VOLUME_NOTE%
if /i "%PROFILE%"=="tuned" echo %CACHE_NOTE%
if /i "%PROFILE%"=="tuned" echo Sampling guard: min-p %MIN_P%, presence penalty %PRESENCE%  (NINFER_MIN_P, NINFER_PRESENCE_PENALTY; "default" = registered preset)
if /i "%PROFILE%"=="tuned" if defined LOOP_NOTE echo %LOOP_NOTE%
rem GRAFT_ARGS carries literal embedded quotes (--graft "godmode=<path>"), so re-quoting it for a
rem string comparison here garbles the quoting and breaks the if statement. `defined` sidesteps
rem that: it tests the variable directly, with no substitution.
if defined GRAFT_ARGS echo Graft: godmode = %GRAFT_FILE%
if defined CHAT_TEMPLATE_ARGS echo Chat template: "%NINFER_CHAT_TEMPLATE%"
if not "%HINT%"=="" echo %HINT%
echo API: http://%HOST%:%PORT%/v1
echo.

rem --host-context-mib 8192 really pins 8 GiB of host RAM here. Earlier builds clamped it on Windows
rem to (free VRAM - 1 GiB) / 2, which left ~4.6 GiB or less; that clamp was removed after measuring
rem that pinning no longer tracks free VRAM on current drivers. See
rem docs\maintainer\launcher-profiles.md.
rem Not a parenthesised block: NINFER_CHAT_TEMPLATE (like NINFER_GRAFT_DIR) is an arbitrary local
rem path and may contain ")" (e.g. "C:\templates\customer (v2)\chat.jinja"). Expanded inside a
rem "( ... )" command group, that character can be taken as the group's own closing paren and
rem break the batch parse, quoting notwithstanding -- so this invocation runs unparenthesised, with
rem goto standing in for the logged/unlogged branch instead.
if /i "%PROFILE%"=="tuned" goto :launch_logged
"%SERVER%" "%MODEL%" --host %HOST% --port %PORT% %PROFILE_ARGS% %NGRAM_VOLUME_ARGS% %STORE_ARGS% %GRAFT_ARGS% %CHAT_TEMPLATE_ARGS%
endlocal
exit /b %ERRORLEVEL%

:launch_logged
rem Run the server with its output shown and kept, so a failure the launcher can explain or retry is
rem told apart from any other. Only a failed host pin retries; a crash or a bad artifact does not.
rem The log is written as ASCII on purpose: Tee-Object writes UTF-16, which findstr cannot search.
set "SERVER_LOG=%TEMP%\ninfer-run-%RANDOM%%RANDOM%.log"
"%SERVER%" "%MODEL%" --host %HOST% --port %PORT% %PROFILE_ARGS% %NGRAM_VOLUME_ARGS% %STORE_ARGS% %GRAFT_ARGS% %CHAT_TEMPLATE_ARGS% 2>&1 | powershell -NoProfile -Command "$input | ForEach-Object { $_; Add-Content -LiteralPath '%SERVER_LOG%' -Value $_ -Encoding Ascii }"
findstr /c:"not enough free GPU memory" "%SERVER_LOG%" >nul 2>&1
if not errorlevel 1 goto :memory_advice
if not "%HOST_RETRY%"=="1" goto :server_done
if "%HOST_CONTEXT_MIB%"=="0" goto :server_done
findstr /c:"cudaMallocHost failed" "%SERVER_LOG%" >nul 2>&1
if errorlevel 1 goto :server_done
set "NEXT_HOST_MIB=0"
if %HOST_CONTEXT_MIB% GTR 2048 set /a NEXT_HOST_MIB=HOST_CONTEXT_MIB/2
echo.
echo Could not pin %HOST_CONTEXT_MIB% MiB of host RAM for the context cache. Retrying with %NEXT_HOST_MIB% MiB.
echo Set NINFER_HOST_CONTEXT_MIB to choose your own, or NINFER_FALLBACK=off to fail instead.
echo.
del "%SERVER_LOG%" >nul 2>&1
set "NINFER_HOST_CONTEXT_MIB=%NEXT_HOST_MIB%"
goto :model_known

:memory_advice
echo.
echo %TITLE% could not start: the graphics card does not have enough free memory, even for a
echo reduced context. Close other programs that use the GPU (games, video, other AI tools)
echo and run this again. `nvidia-smi` lists what is using it.

:server_done
rem The pipe hides the server's own exit status, so report failure from what it logged.
findstr /c:"FATAL" "%SERVER_LOG%" >nul 2>&1
set "SERVER_STATUS=0"
if not errorlevel 1 set "SERVER_STATUS=1"
del "%SERVER_LOG%" >nul 2>&1
endlocal & exit /b %SERVER_STATUS%

:usage
echo usage: run.bat ^<model^> [profile]
echo   qwen38-27b       tuned (default), int8, c8   (recommended)
echo   qwen36-35b-a3b   tuned (default)
echo   qwen38-flash-next tuned (default)   (preview)
exit /b 0
