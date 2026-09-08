{
  description = "NInfer-3090: SM86 (RTX 3090) inference engine for Qwen3.8-27B and Qwen3.6";

  # Pin the nixpkgs revision currently used on this machine (CUDA 12.9 /
  # cuda-merged-12.9, gcc13Stdenv, ffmpeg 6.1). CUDA must be >= 12.8 and the
  # nvcc host compiler must stay within CUDA 12.9's supported GCC range (<= 14),
  # so the build pins gcc13 instead of following the default (gcc15) stdenv.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/241313f4e8e508cb9b13278c2b0fa25b9ca27163";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # NVIDIA CUDA toolkit components
      };

      cuda = pkgs.cudaPackages.cudatoolkit; # cuda-merged-12.9 (nvcc >= 12.8)

      ninfer = pkgs.gcc13Stdenv.mkDerivation {
        pname = "ninfer-3090";
        version = "0.6.1";
        src = self;

        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          cuda
        ];

        buildInputs = [
          pkgs.ffmpeg_6-full # libavformat60/avcodec60/avutil58/swscale7, as validated in the container
          pkgs.curl
        ];

        # The project hard-requires CMAKE_CUDA_ARCHITECTURES=86 (see
        # CMakeLists.txt) and CUDA >= 12.8. gcc13 matches the upstream-validated
        # Linux toolchain and stays inside nvcc 12.9's supported host-compiler
        # range. CC/CXX come from gcc13Stdenv, so CUDA host code uses the same
        # compiler as the C++ side.
        cmakeFlags = [
          "-DCMAKE_CUDA_ARCHITECTURES=86"
          "-DCMAKE_CUDA_COMPILER=${cuda}/bin/nvcc"
          "-DCUDAToolkit_ROOT=${cuda}"
          "-DCMAKE_BUILD_TYPE=Release"
          "-DNINFER_BUILD_APPS=ON"
          "-DBUILD_TESTING=OFF"
          "-DNINFER_BUILD_BENCHMARKS=OFF"
        ];

        # Ninja job pool for single CUDA device-link slot.
        enableParallelBuilding = true;

        # nixpkgs' cmake helper builds in-source by default, so the executables
        # land directly under apps/.
        installPhase = ''
          runHook preInstall
          install -Dm755 apps/ninfer $out/bin/ninfer
          install -Dm755 apps/ninfer-serve $out/bin/ninfer-serve
          runHook postInstall
        '';

        meta = with pkgs.lib; {
          description = "SM86 CUDA inference engine for Qwen3.8-27B / Qwen3.6 on one RTX 3090";
          homepage = "https://github.com/Don-Chad/ninfer-3090";
          license = licenses.asl20;
          platforms = [ "x86_64-linux" ];
          maintainers = [ ];
        };
      };

      # One resumable downloader per registered artifact, mirroring
      # scripts/download-*.sh. NINFER_MODEL_DIR overrides the target directory.
      mkDownload =
        {
          name,
          filename,
          url,
          description,
          revision ? null,
          expectedSize ? null,
          expectedSha256 ? null,
        }:
        pkgs.writeShellScriptBin name ''
          set -euo pipefail
          expected_size='${if expectedSize == null then "" else toString expectedSize}'
          expected_sha256='${if expectedSha256 == null then "" else expectedSha256}'
          stage='${if revision == null then "latest" else revision}'
          resumable=${if revision == null then "0" else "1"}

          model_dir=''${NINFER_MODEL_DIR:-$HOME/models}
          model="$model_dir/${filename}"

          # curl -C - resumes by appending at the current file length, without checking what wrote
          # those bytes. Downloading straight onto "$model" therefore splices a leftover partial
          # from one revision into another whenever a pin changes -- a file of plausible size that
          # is corrupt throughout. Staging under a name that carries the revision means a resume
          # can only ever continue the same artifact -- but an unpinned URL (revision == null,
          # resolving whatever upstream currently calls "main") has no immutable name to stage
          # under: "main" can move between two invocations of this same command, so a leftover
          # ".latest.part" could belong to an older "main" than the one this run would fetch.
          # Unpinned entries therefore never resume: any existing partial is discarded first and
          # the download restarts from zero, at the cost of resumability.
          part="$model.$stage.part"
          mkdir -p "$model_dir"

          if [ -n "$expected_size" ] && [ -f "$model" ] &&
             [ "$(wc -c < "$model" | tr -d '[:space:]')" = "$expected_size" ]; then
            echo "Model already present: $model"
            exit 0
          fi

          if [ "$resumable" = "1" ]; then
            echo "Downloading ${description} to $model (resumable)..."
            ${pkgs.curl}/bin/curl -L -C - --fail --output "$part" '${url}'
          else
            echo "Downloading ${description} to $model (unpinned URL, not resumable)..."
            rm -f -- "$part"
            ${pkgs.curl}/bin/curl -L --fail --output "$part" '${url}'
          fi

          if [ -n "$expected_size" ]; then
            actual_size="$(wc -c < "$part" | tr -d '[:space:]')"
            if [ "$actual_size" != "$expected_size" ]; then
              echo "Expected $expected_size bytes, got $actual_size. Delete $part and retry." >&2
              exit 1
            fi
          fi

          # Set NINFER_SKIP_SHA256=1 to skip: it costs a full re-read of the artifact. The size
          # check above already rejects a truncated or spliced file.
          if [ -n "$expected_sha256" ] && [ "''${NINFER_SKIP_SHA256:-0}" != '1' ]; then
            actual_sha256="$(sha256sum -- "$part" | cut -d' ' -f1)"
            if [ "$actual_sha256" != "$expected_sha256" ]; then
              echo "Checksum mismatch (expected $expected_sha256, got $actual_sha256)." >&2
              echo "Delete $part and retry." >&2
              exit 1
            fi
          fi

          mv -f -- "$part" "$model"
          echo "Model ready: $model"
        '';

      # Qwen3.8-27B (official artifact; validated at C1/C2/C4/C8 on RTX 3090).
      download-qwen38-27b = mkDownload {
        name = "download-qwen38-27b";
        filename = "qwen3_8_27b.ninfer";
        url = "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer";
        description = "Qwen3.8-27B NInfer model";
      };

      # Qwen3.6-27B (groupwise artifact), pinned to match scripts/download-qwen36-27b.sh.
      download-qwen36-27b = mkDownload {
        name = "download-qwen36-27b";
        filename = "qwen3_6_27b.ninfer";
        revision = "faaa0c140d0a92743872256a8b78a954b3984018";
        url = "https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/faaa0c140d0a92743872256a8b78a954b3984018/qwen3_6_27b.ninfer";
        expectedSize = 17495365888;
        expectedSha256 = "7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b";
        description = "Qwen3.6-27B NInfer model";
      };

      # Qwen3.6-35B-A3B, pinned to match scripts/download-qwen36-35b-a3b.sh. 560f227e is the
      # measured 24 GB profile *and* carries the DFlash bundle; the older c8b8c1c0 pin predates
      # DFlash, so an artifact fetched with it cannot run --spec dflash.
      download-qwen36-35b = mkDownload {
        name = "download-qwen36-35b";
        filename = "qwen3_6_35b_a3b.ninfer";
        revision = "560f227e5a7104756d1a108201a8aa75654ea688";
        url = "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/560f227e5a7104756d1a108201a8aa75654ea688/qwen3_6_35b_a3b.ninfer";
        expectedSize = 22783246080;
        expectedSha256 = "1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2";
        description = "Qwen3.6-35B-A3B (pinned, with DFlash) model";
      };

      # Whatever the repository currently calls main. Kept as an escape hatch for trying a newer
      # upstream artifact, so it is deliberately unpinned and deliberately writes to its own
      # filename: it is not the measured profile and it is not what the tests expect. Now that
      # download-qwen36-35b carries DFlash, this no longer exists to supply it.
      download-qwen36-35b-v2 = mkDownload {
        name = "download-qwen36-35b-v2";
        filename = "qwen3_6_35b_a3b_v2.ninfer";
        url = "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/main/qwen3_6_35b_a3b.ninfer";
        description = "Qwen3.6-35B-A3B upstream main (unpinned) model";
      };
    in
    {
      packages.${system} = {
        default = ninfer;
        inherit ninfer;
        inherit download-qwen38-27b download-qwen36-27b download-qwen36-35b download-qwen36-35b-v2;
      };

      apps.${system} = {
        default = {
          type = "app";
          program = "${ninfer}/bin/ninfer-serve";
          meta.description = "NInfer-3090 OpenAI/Anthropic HTTP server";
        };
        cli = {
          type = "app";
          program = "${ninfer}/bin/ninfer";
          meta.description = "NInfer-3090 one-shot CLI generation";
        };
        serve = {
          type = "app";
          program = "${ninfer}/bin/ninfer-serve";
          meta.description = "NInfer-3090 OpenAI/Anthropic HTTP server";
        };
        download-qwen38-27b = {
          type = "app";
          program = "${download-qwen38-27b}/bin/download-qwen38-27b";
          meta.description = "Download the Qwen3.8-27B .ninfer artifact (17 GB)";
        };
        download-qwen36-27b = {
          type = "app";
          program = "${download-qwen36-27b}/bin/download-qwen36-27b";
          meta.description = "Download the Qwen3.6-27B .ninfer artifact (17 GB)";
        };
        download-qwen36-35b = {
          type = "app";
          program = "${download-qwen36-35b}/bin/download-qwen36-35b";
          meta.description = "Download the Qwen3.6-35B-A3B compact v1 .ninfer artifact (21 GB)";
        };
        download-qwen36-35b-v2 = {
          type = "app";
          program = "${download-qwen36-35b-v2}/bin/download-qwen36-35b-v2";
          meta.description = "Download the Qwen3.6-35B-A3B upstream v2 .ninfer artifact (21 GB)";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        inputsFrom = [ ninfer ];
        # Make the CUDA runtime and the NVIDIA driver's libcuda discoverable for
        # ad-hoc testing of store-built binaries from inside the shell.
        shellHook = ''
          export CUDA_PATH=${cuda}
          export LD_LIBRARY_PATH=${cuda}/lib:$LD_LIBRARY_PATH
          # libcuda.so.1 ships with the NVIDIA driver on NixOS
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
          fi
          echo "NInfer-3090 development shell (CUDA 12.9, sm_86)"
          echo "  nix build                                -> build ninfer + ninfer-serve"
          echo "  nix run .#serve -- <serve args>          -> run the HTTP server"
          echo "  nix run .#download-qwen38-27b            -> Qwen3.8-27B artifact (17 GB)"
          echo "  nix run .#download-qwen36-27b            -> Qwen3.6-27B artifact"
          echo "  nix run .#download-qwen36-35b           -> Qwen3.6-35B-A3B compact v1 (21 GB)"
          echo "  nix run .#download-qwen36-35b-v2       -> Qwen3.6-35B-A3B upstream v2 (21 GB)"
        '';
      };
    };
}