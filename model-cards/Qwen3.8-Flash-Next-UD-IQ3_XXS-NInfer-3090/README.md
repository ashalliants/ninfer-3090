---
library_name: ninfer
pipeline_tag: text-generation
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
  - conversational
  - cuda
  - rtx-3090
  - gguf-derived
  - unsloth
---

# Qwen3.8-Flash-Next, Unsloth UD-IQ3_XXS, for NInfer on the RTX 3090

> **Maintained at [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090)**: the engine, converter and release builds these artifacts are made and tested with. Releases, issues and updates live there, and a star helps others find it.

This is the **3-bit build** of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (48 layers,
512 routed experts) for the RTX 3090 fork of NInfer,
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090). It is Unsloth's
[Dynamic v3.1 `UD-IQ3_XXS` GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) rewritten into
NInfer's `.ninfer` container, plus the MTP drafter. It sits beside the
[IQ2_XS build](https://huggingface.co/WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090) of the same model so that
you can pick between them. Like that one it is a **preview**: Windows launcher, one request at a time
(two lanes opt-in), 64 GB of RAM. It is not a Transformers checkpoint, Safetensors distribution or
GGUF file, and it needs this fork's executables.

## Which build should I use?

| | **IQ2_XS** (the first build) | **UD-IQ3_XXS** (this one) |
|---|---|---|
| Source | ISTA DASLab GSQ-RCO GGUF | Unsloth Dynamic v3.1 GGUF |
| Routed experts, average | 2.35 bits per weight | 3.01 bits per weight |
| Text quality (perplexity, 5,114 fixed positions, lower is better) | 6.315 | **5.859 (7.2% lower)** |
| Decode with MTP, 1K / 8K / 30K prompt | **115 / 111 / 106 tok/s** | 61 / 62 / 64 tok/s |
| Decode without MTP | **84 / 86 / 83 tok/s** | 46 / 55 / 56 tok/s |
| Prefill, 8K / 30K prompt | **2,225 / 2,293 tok/s** | 894 / 905 tok/s |
| Images | yes | **no** (see Limits) |
| RAM to start | about 40 GiB available | about 42 GiB available |
| Disk | 41 GB model + 29 GB volume | 57 GB model + the same 29 GB volume |

Pick **IQ2_XS** for speed and images: it decodes 1.5-1.9 times and prefills 1.8-2.6 times as fast. Pick
**UD-IQ3_XXS** when you want the best text quality this card can hold and can live with about half
the speed. Both run from one 29 GB n-gram volume, which you download once.

## What the conversion does

Every value is stored as the source holds it, with one exception described below.

- **Text weights** (from `UD-IQ3_XXS`): block tensors keep their GGML blocks byte for byte. The routed
  experts' gate and up are IQ2_S, IQ3_XXS or IQ4_XS and their down Q2_0 or IQ4_NL, chosen per layer by
  Unsloth; the dense projections, the token embedding and the output head are Q6_K, with one layer's
  GDN qkv and z and shared-expert gate and up in Q8_0 and every shared-expert down in Q8_0. Each layer's
  512 experts become one bank in `ggml_expert_record_v1`, one record per expert holding its gate, up and
  down blocks unchanged, so an expert moves between host RAM and the GPU as one byte range.
- **The one rounding.** The model reads some tensors as BF16: the hyper-connection mixers' `down` and
  `up` matrices and the per-layer-embedding `key` and `value` projections, which Unsloth stores as
  Q8_0, and the router, the shared-expert score, the GDN `a`/`b` projections and the hyper-connection
  `inject` rows, which it stores as F32. The F32 ones already hold BF16 words, and the converter refuses
  any that do not. The Q8_0 ones are decoded exactly to FP32 and rounded once to nearest even:
  196 tensors, 642,735,924 values, each changed by at most half a BF16 step (a relative error
  of at most 2^-8). Nothing else is rounded.
- **MTP drafter** (component `mtp`, from Strata's `mtp-q2_0.gguf`): the same drafter as the IQ2_XS build.
- **Vision tower** (component `vision`, from Unsloth's `mmproj-BF16.gguf`): carried in the file as in the
  IQ2_XS build, but this build cannot load it (see Limits).
- **N-gram volume**: the per-layer embedding (PLE) n-gram table, 320,001,536 IQ4_NL rows. It is **not in this
  repository**: Unsloth's table is the same table as DASLab's, row for row, so this artifact is bound to
  the volume of the [IQ2_XS repository](https://huggingface.co/WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090)
  (`qwen4exp-iq2xs-mtp-vision.ninfer.ngram`, volume id `1604acb23af5cb05f001e2ce3a8c0f12`) and the
  server refuses any other. The downloader fetches it once for both builds.

An independent checker (`tests/convert/qwen4_exp_verify.py`, its own mapping and its own integer
rounding rather than the converter's) compares every stored object and every volume row against the
source GGUFs. Run on the published files (SHA-256 values below), it found all 1,356 tensor objects exact: the
48 text expert banks and the drafter's bank unpacked to their GGUF tensors byte for byte, 499 text, 28 drafter
and 333 vision objects identical, 120 narrowed exactly (the q/k norms and GDN tensors stored as BF16 words), 132
F32 objects narrowed exactly to BF16, and the 195 objects rounded from Q8_0 equal to the checker's own integer
rounding of the GGUF's values. All 320,001,536 volume rows equal the GGUF's. The source GGUF files are the ones
Hugging Face lists, SHA-256 for SHA-256.

## Files

| File | Bytes | SHA-256 | Read |
|---|---:|---|---|
| `qwen4exp-ud-iq3xxs-mtp-vision.ninfer` (the model) | 32,000,000,000 | `0edec1ba7ba56e3cf0a0a2b3bfa990a7a645243c649265e507cdfa78c37a2fe6` | once, at load |
| `qwen4exp-ud-iq3xxs-mtp-vision.ninfer.part-0001` (its continuation) | 25,087,457,792 | `f971f41af085d3e40cd3ef6658f34d09871f3e7145d328c18df1ec3010a021f7` | once, at load |
| `qwen4exp-ud-iq3xxs-mtp-vision.ninfer.conversion.json` (conversion report) | 504,011 | `8b215e892702678c3f306c346c13fdf049ab0b4ecfd6d425561781582351c273` | not needed to run |
| `qwen4exp-iq2xs-mtp-vision.ninfer.ngram` (n-gram volume, **from the IQ2_XS repository**) | 29,127,258,112 | `7623459b2385eef4505891c2ff133b3dd8ad4ecd0ad334c901878f3fa1d1b051` | on every prompt |

The model and its part are one artifact (id `fa9dd32ccdad435e8123ab8fa8418e64`, components `text`, `mtp`
and `vision`, 1,362 objects): both must sit in the same folder under these names.
`artifact-manifest.json` lists the same files, the sources and the inventory.

```bash
sha256sum --check SHA256SUMS
```

## Hardware

- **GPU:** RTX 3090 (24 GB). The dense weights, the KV cache and a cache of the most-used experts
  (about 5,900-6,400 of the 24,576) share the card.
- **RAM: 64 GB.** The experts do not all fit beside Windows, so this build runs with
  `--partial-ram-residency` (the launcher turns it on): the experts the VRAM cache does not hold are
  pinned in host RAM, 36.5 GiB of them plus 4.3 GiB of other pins. Before pinning, the server checks that
  they and 2 GiB for Windows fit in the memory available now, and refuses with the shortfall if not.
  Plan on about 42 GiB available at the start, 46 GiB with the default 4 GiB prefix cache, which the
  server shrinks to what it can spare. On the 64 GB test machine that was **tight with a browser open**:
  39.8 GiB were available then, 1.5 GiB short, and 49.0 GiB with the browser and other programs closed.
- **Disk:** about 86 GB: the model (57.1 GB in two files) and the 29.1 GB n-gram volume.
- **OS:** Windows. The `run.bat qwen38-flash-next-iq3` launcher is the supported way to serve it.

## Measurements

RTX 3090 (24 GB, WDDM, desktop attached), CUDA 12.8, Ryzen 9 5950X, 64 GB DDR4-2133, Windows 11,
model and n-gram volume both on one SATA SSD (Samsung 860 QVO). Both builds ran the launcher's
`qwen38-flash-next` profile, INT8 KV, 32,768 tokens of context, one lane (the plain rows with a copy of the
profile that leaves out its two MTP-only options, `mtp-joint-acceptance` and `gdn-async-commit`, which
the server refuses without `--spec mtp`), prefix reuse bypassed (a fresh prompt every request), after 12 warm-up requests had filled the expert cache. Each cell is the
median of 5 requests [min-max], 512 generated tokens; prefill and decode rates are the server's own
per-request figures. The IQ2_XS rows reproduce the figures in its card (decode 81 / 80 / 78 plain and
109 / 111 / 113 with MTP).

**Decode**, tok/s after a prompt of about 1K / 8K / 30K tokens:

| | 1K | 8K | 30K |
|---|---:|---:|---:|
| IQ2_XS, plain | 84.2 [82.2-88.5] | 86.3 [82.7-87.3] | 83.3 [82.4-86.9] |
| IQ2_XS, MTP (up to 3 drafts) | 115.4 [98.2-129.0] | 110.8 [99.0-112.5] | 106.0 [101.1-111.8] |
| **UD-IQ3_XXS, plain** | 46.0 [44.0-56.1] | 54.6 [49.1-61.2] | 56.2 [52.1-56.7] |
| **UD-IQ3_XXS, MTP (up to 3 drafts)** | 61.0 [57.7-65.0] | 62.4 [61.0-69.1] | 63.8 [60.4-65.5] |
| UD-IQ3_XXS / IQ2_XS, plain | 0.55 | 0.63 | 0.67 |
| UD-IQ3_XXS / IQ2_XS, MTP | 0.53 | 0.56 | 0.60 |

**Prefill**, tok/s, the same prompts (about 940 / 6,330 / 24,830 tokens):

| | 1K | 8K | 30K |
|---|---:|---:|---:|
| IQ2_XS | 1,142 [1,035-1,228] | 2,225 [2,085-2,260] | 2,293 [2,141-2,350] |
| **UD-IQ3_XXS** | 650 [621-690] | 894 [878-935] | 905 [891-926] |
| UD-IQ3_XXS / IQ2_XS | 0.57 | 0.40 | 0.39 |

MTP gives a smaller gain here than on IQ2_XS (1.14-1.33 times against 1.27-1.37). Why this build is
slower is not measured; the likeliest causes are its smaller expert cache (about 6,000 experts in VRAM
against 11,000-12,500) and heavier dense weights (4.2 GiB against 3.5 GiB), and that
`--partial-ram-residency` prefills without the expert stream. The mode's own decode and prefill were
not tuned.

**Startup.** Loading took 2 min 25 s to 2 min 31 s (the weights 11 s, then reading about 6,000 experts
into VRAM and pinning the rest). The first requests of a process are slower while the expert cache fills: the first four of
a fresh server decoded at 21, 32, 45 and 38 tok/s.

**Quality.** Teacher-forced log-probabilities of the repository's six fixed test texts (code, a
document and a chat each, from two sets, 5,114 predicted positions; the fixtures of
`tests/fixtures/qwen4_exp`), scored by NInfer for both builds on the same tokens. ΔNLL is
UD-IQ3_XXS − IQ2_XS in nats, ± its paired standard error; negative is better:

| Set | Positions | ppl IQ2_XS | ppl UD-IQ3_XXS | ΔNLL | top-1 next token IQ2_XS / UD-IQ3_XXS |
|---|---:|---:|---:|---:|---:|
| Strata's prompts, code | 511 | 1.5225 | 1.4846 | −0.025 ± 0.019 | 91.0% / 91.6% |
| Strata's prompts, document | 1,023 | 21.4595 | 19.4280 | −0.100 ± 0.025 | 43.9% / 45.1% |
| Strata's prompts, chat | 1,023 | 6.5936 | 5.7525 | −0.137 ± 0.028 | 55.6% / 57.3% |
| Infernix's prompts, code | 511 | 1.9616 | 1.8849 | −0.040 ± 0.026 | 84.5% / 84.1% |
| Infernix's prompts, document | 1,023 | 7.8517 | 7.1901 | −0.088 ± 0.026 | 54.9% / 57.2% |
| Infernix's prompts, chat | 1,023 | 5.2231 | 5.1304 | −0.018 ± 0.034 | 63.1% / 63.1% |
| **All six** | **5,114** | **6.3147** | **5.8591** | **−0.075 ± 0.012** | **61.1% / 62.1%** |

The gain is about six standard errors overall and is clear on the documents and one chat; on the two code
texts and the other chat it is inside the noise. The IQ2_XS perplexities equal the ones published for it
to four digits, which checks the method. This compares two quantizations against each other on text
that is probably partly in the model's training data; there is no BF16 or Q8_0 reference here, so it
ranks the builds and says nothing of either one's distance from the original model.

**MTP and plain decoding give the same output.** Six prompts at temperature 0, each run once with and
once without MTP: the four that finished inside the 400-token limit (a sum, a Python function, an
explanation, a translation) are identical text, one of them 313 tokens long, and the token counts are
identical on all six; the other two (long document prompts) ended mid-thinking, so only their counts were
compared. That is a smaller check than the IQ2_XS build's 108 requests.

## Run it

Build [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090), then:

```text
scripts\download-model.bat qwen38-flash-next-iq3
scripts\run.bat qwen38-flash-next-iq3
```

The downloader fetches the files above at a pinned revision, checks each one's size and SHA-256 before
keeping it, and resumes an interrupted download. It asks where the n-gram volume should go (default:
beside the model); if you already have the IQ2_XS build's volume beside the model it is reused. If the
volume is on another drive, tell the launcher:

```text
set NINFER_NGRAM_VOLUME=D:\ninfer-models\qwen4exp-iq2xs-mtp-vision.ninfer.ngram
scripts\run.bat qwen38-flash-next-iq3
```

or start the server directly with `--ngram-volume FILE --partial-ram-residency --spec mtp --draft-tokens 3`.
The launcher serves one request at a time by default (`NINFER_CONCURRENCY=2` runs two lanes) at 32,768
tokens with INT8 KV and MTP; `NINFER_CONTEXT` raises the context and `NINFER_SPEC=none` turns MTP off.

## Provenance

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.8-Flash-Next` (Qwen Community License 1.0) |
| Text weights | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `f5d175de2cf7f720d45d6064019a81fcf056e986`: `UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf` (10,946,618 bytes, metadata only, `9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915`), `-00002-of-00004.gguf` (49,946,027,488 bytes, `aacaff3bdb9032c2c3be428167930809735f65d3cfb204eba07d85e421e7ddf4`), `-00003-of-00004.gguf` (4,869,325,888 bytes, `ef8f14e790ab471b363f146d5e8664eefbaf03e5776f317a1fb96a2bc7707b07`) and `-00004-of-00004.gguf` (28,800,138,432 bytes, the n-gram table, `5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36`) |
| Vision tower | the same repository's `mmproj-BF16.gguf` (907,542,944 bytes, `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`) |
| MTP drafter | `mtp-q2_0.gguf` as installed by Strata 0.1.38 (889,017,216 bytes, `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d`), the same file as the IQ2_XS build |
| Recipe | `qwen4_exp_gguf` |
| Converter | `python -m tools.convert --model Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf --mtp-gguf mtp-q2_0.gguf --mmproj mmproj-BF16.gguf --recipe qwen4_exp_gguf --device cpu --out qwen4exp-ud-iq3xxs-mtp-vision.ninfer --ngram-reuse qwen4exp-iq2xs.ninfer.ngram` with the converter of fork commit `848cbe3f8` (PR #481, which merged it into master) |

The GGUF hashes are those Hugging Face reports for that revision, and the local copies that were converted
have the same SHA-256 values. Unsloth's repository also holds an importance matrix
(`imatrix_unsloth.gguf_file`); this conversion does not use it, because it re-quantizes nothing: the
quantization is Unsloth's own. Model code adapted from [Infernix](https://github.com/wallawalla47/infernix)
(Apache-2.0) is credited in the fork's README.

## Limits

- **Slower than the IQ2_XS build**, by a lot (tables above): it is the quality option, not the speed option.
- **No images.** The file carries the vision tower, but `--partial-ram-residency`, which this build needs,
  refuses `--vision`. Use the IQ2_XS build for images.
- **`--partial-ram-residency` is new.** Its decode with write-backs and its startup time were measured
  here for the first time; they are not tuned, and prefill under it does not use the expert stream.
- **RAM is tight on a 64 GB machine.** Close browsers before starting if the server reports a shortfall.
- **One request at a time by default**, INT8 KV only, as for the IQ2_XS build. Two lanes are untested here.
- **Quality evidence is narrow.** Six short fixed texts, two builds compared with each other, no
  full-precision reference, and no task benchmarks run. Unsloth and DASLab publish their own
  evaluations of their quantizations; neither covers this container.
- **Rounded tensors.** 642,735,924 values of 196 tensors (the Q8_0 mixers and PLE projections) were rounded
  to BF16 once (above). Their effect on output was not measured separately.
- **Not verified:** machines with less than 64 GB of RAM; Linux; a full download from this repository
  followed by a run on a second machine.

## License

The weights are Qwen's. Qwen3.8-Flash-Next is released under the **Qwen Community License 1.0**,
included here as `LICENSE`, and this conversion is distributed under the same license; its notice
must be kept with any copy. The license allows use, modification and redistribution, and asks for a
separate license from Qwen for commercial "Model as a Service" or "AI Work Assistant" businesses;
read `LICENSE` for the exact terms.

The conversion's sources declare their licenses differently, so they are listed plainly: the Unsloth
GGUF repository declares `qwen-community-1.0` (license `other`); Strata's `mtp-q2_0.gguf` carries no
license metadata and holds tensors taken from Qwen's checkpoint. A quantization does not change the
license of the weights it encodes, so this artifact follows the base model's license.

---

Built and maintained by [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090). Development is sponsored by [NeverMetered](https://nevermetered.com/?ref=ninfer-3090-hf), hosted Qwen3.8-27B for coding agents.
