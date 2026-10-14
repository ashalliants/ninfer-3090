---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - moe
  - multimodal
  - conversational
  - cuda
  - rtx-3090
  - gguf-derived
---

# Qwen3.8-Flash-Next for NInfer on the RTX 3090

This model card is the version-controlled source for
[WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090](https://huggingface.co/WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090).

The repository holds [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (48 layers,
512 routed experts) for the RTX 3090 fork of NInfer,
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090). It is the
[ISTA DASLab GSQ-RCO IQ2_XS GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
rewritten into NInfer's `.ninfer` container without re-quantizing anything, plus the MTP drafter and
the vision tower. It is not a Transformers checkpoint, Safetensors distribution or GGUF file, and it
needs this fork's executables. Support is a **preview**: one request at a time (two lanes opt-in), Windows launcher only.
It takes text and images (below).

## What the conversion does

Every value is stored exactly as the sources hold it; nothing is rounded again.

- **Text weights** (from the GSQ-RCO IQ2_XS GGUF): block tensors keep their GGML blocks byte for
  byte (routed experts' gate and up in IQ2_S, IQ2_XXS or IQ1_M and their down in Q2_0; dense
  weights in IQ4_XS, IQ3_S, IQ4_NL, Q6_K, Q8_0 and Q2_0, as GSQ-RCO allocated them), BF16 and FP32 keep
  their words, and the one F16 tensor widens exactly to FP32.
  A few direct tensors are stored in the form their kernel reads, still without rounding (the q/k
  norms as the BF16 `gamma` of the GGUF's `1 + gamma`; the converter refuses any value without an
  exact form).
- **Routed experts**: each layer's 512 experts become one bank in `ggml_expert_record_v1`, one
  record per expert holding its gate, up and down blocks unchanged, so an expert moves between host
  RAM and the GPU as one byte range. Unpacking the records gives back the GGUF tensors exactly.
- **MTP drafter** (component `mtp`, from Strata's `mtp-q2_0.gguf`): BF16 matrices keep their words;
  its 512 routed experts keep their Q2_0 blocks.
- **Vision tower** (component `vision`, from the GGUF repository's `mmproj-Qwen3.8-Flash-Next-BF16.gguf`):
  Qwen3.5's vision encoder geometry, every value stored as BF16 (the mmproj's F32 tensors hold
  BF16-exact values).
- **N-gram volume**: the per-layer embedding (PLE) n-gram table, 320,001,536 IQ4_NL rows, is not in
  the `.ninfer`. It is a separate file in Infernix's `NINFERNG` format (version 2, 4 KiB blocks, each
  row's GGML blocks unchanged), read a few rows per token. The artifact records the volume's geometry
  and a 16-byte id, and the server refuses a volume with another id.

An independent checker (`tests/convert/qwen4_exp_verify.py`, its own mapping rather than the
converter's) compares every stored object and every volume row against the three GGUFs. Run on the
published files (the SHA-256 values below), it found all 1,356 tensor objects byte-exact: 825 text
objects identical, 120 narrowed and 1 widened exactly, the 48 expert banks, the drafter's 28 objects
and bank, and the 333 vision objects. All 320,001,536 volume rows equal the GGUF's, and the volume's
header equals the geometry and id the artifact records.

## Files

| File | Bytes | SHA-256 | Read |
|---|---:|---|---|
| `qwen4exp-iq2xs-mtp-vision.ninfer` (the model) | 32,000,000,000 | `87c871e6cc1b6392e387104e8dfb159283a8a64247f669fbf33355b19b769ef4` | once, at load |
| `qwen4exp-iq2xs-mtp-vision.ninfer.part-0001` (its continuation) | 9,012,140,032 | `1798c63c2f74a05cac71a609701f34368f0ba1e1eeebbf1b7641e83d0dd8d811` | once, at load |
| `qwen4exp-iq2xs-mtp-vision.ninfer.ngram` (n-gram volume) | 29,127,258,112 | `7623459b2385eef4505891c2ff133b3dd8ad4ecd0ad334c901878f3fa1d1b051` | on every prompt |
| `qwen4exp-iq2xs-mtp-vision.ninfer.conversion.json` (conversion report) | 527,509 | `73a3525c4582f7459fa3b6684d5b45e15ce6663222908e73a49a278d3ad1c040` | not needed to run |

The model and its part are one artifact (id `a1b2f8c8de7744c1a723735051f7e47c`, components `text`,
`mtp` and `vision`, 1,362 objects): both must sit in the same folder under these names.
`artifact-manifest.json` lists the same files, the sources and the inventory.

The volume's header: `NINFERNG` version 2, 320,001,536 rows of 90 bytes (`ggml_iq4_nl`), 7,111,146
blocks, volume id `1604acb23af5cb05f001e2ce3a8c0f12`. It is named `<artifact>.ngram` because that
is where the server looks by default; it can live elsewhere (below).

```bash
sha256sum --check SHA256SUMS
```

## Hardware

- **GPU:** RTX 3090 (24 GB). The dense weights, the KV cache and a cache of the most-used experts
  (about 11,000-12,500 of the 24,576) share the card.
- **RAM: 64 GB.** All 33.0 GiB of routed experts are pinned in host RAM at startup, and with images
  on (`--vision`, the launcher's default for this file) the 0.84 GiB vision tower too. Before
  pinning, the server checks that the experts, its other allocations and 2 GiB for Windows fit in
  the memory available now, and refuses with the shortfall if not; plan on about 40 GiB free
  (about 2.2 GiB less with `NINFER_VISION=off`). On the 64 GB test machine, 45-50 GiB was available
  at each start.
- **Disk:** about 70 GB: the model (41.0 GB in two files) and the 29.1 GB n-gram volume.
- **Put the n-gram volume on NVMe.** It is read unbuffered on every prompt; the model is read only
  at load. With the model on a SATA SSD, moving only the volume to an NVMe drive changed warm prefill
  from 0.95x / 0.97x / 0.98x Strata's to 1.23x / 1.01x / 1.09x at 4K / 32K / 128K tokens, and the
  spread between prompts almost disappeared (table below). Decode barely moved (+0.8%).
- **OS:** Windows. The `run.bat qwen38-flash-next` launcher is the supported way to serve it.

## Measurements

RTX 3090 (24 GB, WDDM, desktop attached), CUDA 12.8, Ryzen 9 5950X, 64 GB DDR4-2133, Windows 11.
Strata 0.1.38 running the same IQ2_XS GGUF is the reference; its figures are its own runs on this
machine (three per tier). Speeds are warm: the expert cache filled, every request after a process's
first. **The tables were measured on the text-only and text+MTP conversions of the same GGUF**
(`qwen4exp-iq2xs.ninfer`, `qwen4exp-iq2xs-mtp.ninfer`), which share every text weight and the
volume with this artifact. This artifact was then measured against them with images on (four runs,
A B B A, the same prompts; ratios this artifact / the other, mean ± sd), with identical greedy
tokens throughout:

| | 1K | 8K | 32K |
|---|---:|---:|---:|
| Plain decode | 0.9970 ± 0.0022 | 0.9957 ± 0.0026 | 0.9983 ± 0.0021 |
| MTP decode | 1.0008 ± 0.0014 | 0.9959 ± 0.0042 | 0.9994 ± 0.0004 |
| Prefill (4K and 32K) | 0.9976 ± 0.0025 | | 1.0027 ± 0.0017 |

So its text speed is the tables' within half a percent, though plain decode is consistently
0.2-0.4% slower.

**Quality.** Teacher-forced log-probabilities on fixed token ids against Strata 0.1.38 on the same
GGUF (`tools/flash_next/strata_compare.py`; ΔNLL = NInfer − Strata in nats, ± its paired standard
error):

| Set | Positions | ppl NInfer | ppl Strata | ΔNLL | top-1 agreement | KL mean / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Strata's prompts (code, document, chat) | 2,557 | 7.8876 | 7.8371 | +0.0064 ± 0.0027 | 97.1% | 0.0042 / 0.039 |
| Infernix's prompts | 2,557 | 5.0555 | 5.0518 | +0.0007 ± 0.0025 | 97.4% | 0.0041 / 0.038 |

Strata against itself gives 96.5-99.2% top-1 agreement on the same texts. Two figures are more than
two standard errors above zero: the first set overall (+0.0064, 2.4 SE) and its chat text (+0.0158,
3.2 SE); the second set is within 1.2 SE on every text. This compares two runtimes on the same
2-bit weights. What the IQ2_XS quantization itself costs against the BF16 model is DASLab's
measurement, on their model card.

**Decode**, 512 tokens after a prompt of the given length, tok/s, median of 12 requests:

| | 1K | 8K | 32K |
|---|---:|---:|---:|
| Plain | 81.4 | 80.2 | 78.2 |
| MTP, up to 3 drafts per round (the launcher's default) | 109.1 | 111.2 | 113.0 |
| Strata 0.1.38, plain | 71 | 70 | 66 |
| Strata 0.1.38, MTP | 105 | 112 | 107 |

MTP gives the same tokens as plain greedy decoding (checked on 108 requests). It accepts 2.2-2.6
tokens per round. After a 128K-token prompt, decode is 74.0 tok/s plain and 102.6 with MTP.

**Prefill**, warm, tok/s, the same plain artifact on a SATA SSD with the n-gram volume on each drive:

| Prompt | Volume on SATA | Volume on NVMe | Strata 0.1.38 | NVMe / Strata |
|---|---:|---:|---:|---:|
| 4K | 1,825 [1,334-2,128] | 2,345 [2,320-2,393] | 1,912 | 1.23x |
| 32K | 2,332 [1,814-2,372] | 2,422 [2,406-2,444] | 2,397 | 1.01x |
| 128K | 2,135 [1,868-2,345] | 2,372 [2,346-2,400] | 2,184 | 1.09x |

Medians [min-max], n = 12 / 12 / 6 per drive. At 32K the margin over Strata is inside its spread:
call it parity. With MTP on, prefill is 1-4% slower (the drafter computes a cell for every prompt
position and its experts take some of the cache).

**Startup and the first request.** Loading took 93-104 s from a SATA SSD (33.0 GiB pinned,
3.5 GiB on the GPU). The first request in a process with an empty expert cache is much slower:
1,054-1,079 tok/s prefill at 4K from NVMe (635-893 from SATA) against 2,345 warm. The server saves
the expert cache beside the model (`<model>.expert-state`) at shutdown and every ten minutes, so
later starts begin warm. With images on, startup takes 2.2-6.9 s longer: it encodes a blank image so
that the first real one does not pay that one-time setup.

## Images

The vision tower takes images through the Engine and both of the server's routes: OpenAI
`image_url` (data URIs) and Anthropic `image` blocks. The launcher turns it on for this file
(`NINFER_VISION=off` turns it off). Video input is accepted by the route but was not tested.

- **Answers:** five real test images (text in an image, counting, positions, a chart, a scene
  description) were all answered correctly, greedy, through the Engine; six OpenAI requests and one
  Anthropic request through `run.bat qwen38-flash-next` were too.
- **Tokens:** the image token counts follow the Qwen3.5/3.8 processor and chat template, for example
  384 tokens for 768 x 512 and 2,280 for 1920 x 1200. They equal Strata's wherever both make the same
  grid; for 1920 x 1200 llama.cpp caps the image at 1,000 tokens, so that case differs by design.
- **Time** (warm): encoding takes 0.05-0.10 s for a 384-token image and 0.41-0.46 s for a 2,280-token
  one; a whole 426-443-token image prompt prefills in 0.7-1.4 s, a 2,301-token one in 1.5-2.3 s.
- **Memory:** the tower lives in pinned host RAM (0.84 GiB) and uses no VRAM while idle. A request
  borrows its working memory from the expert cache and returns it; 306-976 MiB of VRAM stayed free
  during image requests, and an image in a 41,728-token prompt fit with 0 MiB over budget. Images
  add about 2.2 GiB to the RAM the server reserves at startup.
- **MTP with images:** the same tokens as plain greedy decoding (185 tokens, 88 rounds checked).

**How it differs from Strata.** This tower computes in BF16, as NInfer's Qwen3.5 tower does, where
llama.cpp (Strata's vision path) keeps activations in F32. Against an FP64 reference of the encoder,
the stored-BF16 computation alone is 2-6% off on realistic image sizes, so image embeddings differ
from Strata's by about that much. Of five comparable greedy answers, three were identical to
Strata's and two diverged after 10 and 17-26 tokens (Strata's own two runs of one of those disagree
with each other from token 17, while NInfer's repeat exactly).

**The encoder check is loose.** The tower was qualified against an independent FP64 implementation
of the encoder, reading the stored weights directly. The tolerance first set (normwise 2^-5) failed
on 3 of 10 test images and was loosened twice after measuring, to normwise max(2^-4, twice the error
of BF16 storage alone). Measured errors were 1.3e-2 to 5.5e-2 on grid sizes the processor produces
(2.0e-1 on a tiny 2 x 4 grid it never produces, where BF16 storage alone accounts for 0.17). That
bound catches a wrong weight mapping, order, position or projection (deliberately broken versions
land at 0.69-1.05), not a drift of a few percent; the answers above and the Strata comparison are the
evidence for the rest.

## Run it

Build [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090), then:

```text
scripts\download-model.bat qwen38-flash-next
scripts\run.bat qwen38-flash-next
```

The downloader fetches the three files above at a pinned revision, checks each one's size and
SHA-256 before keeping it, and resumes an interrupted download. It asks where the n-gram volume
should go (default: beside the model) and prints the lines to start the server with. If the volume
is on another drive, tell the launcher:

```text
set NINFER_NGRAM_VOLUME=D:\ninfer-models\qwen4exp-iq2xs-mtp-vision.ninfer.ngram
scripts\run.bat qwen38-flash-next
```

or start the server directly with `--ngram-volume FILE`. The launcher serves one request at a time by default
(`NINFER_CONCURRENCY=2` runs two lanes) at 32,768 tokens with INT8 KV, MTP and images; `NINFER_CONTEXT` raises the context (KV beyond 32,768
tokens is borrowed from the expert cache; 128K was measured), `NINFER_SPEC=none` turns MTP off and
`NINFER_VISION=off` turns images off.

## Provenance

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.8-Flash-Next` (Qwen Community License 1.0) |
| Text weights | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09`: `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` (39,225,954,592 bytes, `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7`) and `-00002-of-00002.gguf` (28,800,138,432 bytes, `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`) |
| Vision tower | the same repository's `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 bytes, `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`) |
| MTP drafter | `mtp-q2_0.gguf` as installed by Strata 0.1.38 (889,017,216 bytes, `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d`): the 31 `mtp.*` tensors of Qwen's BF16 checkpoint, routed experts in Q2_0 |
| Recipe | `qwen4_exp_gguf` |
| Converter | `python -m tools.convert --model Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf --mtp-gguf mtp-q2_0.gguf --mmproj mmproj-Qwen3.8-Flash-Next-BF16.gguf --recipe qwen4_exp_gguf --device cpu --out qwen4exp-iq2xs-mtp-vision.ninfer --ngram-reuse qwen4exp-iq2xs.ninfer.ngram` with the converter of fork commit `06d812b2687022093e95bd8775b9d745893f3098` (the head of the unmerged image-input branch `feat/qwen4exp-vision`, PRs #270-#272; the conversion ran on that converter before the branch was rebased onto master, and the result is checked byte for byte against the sources above) |

The GGUF hashes are those Hugging Face reports for that revision; the local copies that were
converted have the same SHA-256. The MTP file is not published by its own repository, so its hash
is that of the local copy. Model code adapted from [Infernix](https://github.com/wallawalla47/infernix)
(Apache-2.0) is credited in the fork's README.

## Limits

- **MTP missed its target.** The goal was 1.2x Strata's MTP decode at 8K (134 tok/s); it reached
  111-113 tok/s, level with Strata (1.01x at 8K, 1.05x at 1K and 32K). The cost is in verifying each
  draft column: every extra column routes more experts.
- **One request at a time by default.** `NINFER_CONCURRENCY=2` serves two lanes; the second request then
  decodes beside the first (plain, no MTP while both decode) instead of waiting. A burst of eight agent
  requests finished 6.5% sooner with a 19% lower mean time to first token. INT8 KV only.
- **Slow first request** after a start with no saved expert state (above).
- **Needs the fork's image-input change.** At the time of writing, the change that reads this file's
  vision tower and picks it up in the launcher (PRs #270-#272) and the launcher's
  `NINFER_NGRAM_VOLUME` (PR #269) are not yet merged into the fork's master. A build of master
  alone runs this file text-only and without MTP (the launcher chooses MTP by file name).
- **Images:** image embeddings differ from Strata's by about 2-6% (BF16 against F32, above), and the
  encoder's FP64 check is loose. Video and images on more than one lane at once are untested.
- **Not verified:** absolute speeds of this file through the server (its text speed was measured
  only as ratios through the Engine, above); machines with less than 64 GB of RAM; Linux; a full
  download from this repository followed by a run (only partial downloads with resume, and the
  checks of files already present).

## License

The weights are Qwen's. Qwen3.8-Flash-Next is released under the **Qwen Community License 1.0**,
included here as `LICENSE`, and this conversion is distributed under the same license; its notice
must be kept with any copy. The license allows use, modification and redistribution, and asks for a
separate license from Qwen for commercial "Model as a Service" or "AI Work Assistant" businesses;
read `LICENSE` for the exact terms.

The conversion's sources declare their licenses differently, so they are listed plainly: the
DASLab GGUF repository and its IQ2_XS files declare `apache-2.0`; its mmproj declares
`qwen-community-1.0`; Strata's `mtp-q2_0.gguf` carries no license metadata and holds tensors taken
from Qwen's checkpoint. A quantization does not change the license of the weights it encodes, so
this artifact follows the base model's license.
