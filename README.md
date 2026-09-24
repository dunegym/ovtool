# ovtool — A Comprehensive OpenVINO Command-Line Tool

A multi-purpose inference CLI built on [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai), supporting:

- **LLM inference**: one-shot generation + interactive multi-turn chat, streaming output, full sampling controls
- **OpenAI-compatible API server**: expose a converted LLM behind `POST /v1/chat/completions` (incl. SSE streaming), `POST /v1/completions` and `GET /v1/models`
- **Browser WebUI**: pick any converted model, load it on a device and chat / generate images from the browser (`ovtool webui`)
- **Multimodal (VLM) inference**: image + text Q&A (converted LLaVA / Qwen-VL / MiniCPM-V / InternVL models)
- **Text-to-speech (TTS)**: SpeechT5 and Kokoro-82M synthesis to WAV, with speaker/voice selection, language and speed controls
- **Embeddings & reranking**: text vectorization (query/document modes, cosine retrieval ranking) and cross-encoder reranking for RAG pipelines
- **Image generation**: Text2Image and Image2Image with the SD / SDXL / Flux families
- **Device selection & runtime options**: CPU / GPU / NPU / AUTO / HETERO, with pass-through OpenVINO runtime properties
- **Model conversion & quantization**: one-command export of Hugging Face models to OpenVINO IR with INT8 / INT4 weight compression (AWQ supported)

## Setup

```bash
conda create -n openvino-cli python=3.11 -y -c conda-forge --override-channels
conda activate openvino-cli
pip install -e .            # inference deps (openvino / openvino-genai / openvino-tokenizers)
pip install "optimum-intel[openvino]" onnx   # conversion/quantization deps (or pip install -e ".[convert]")
```

After installation the command entry point is `ovtool` (equivalent to `python -m ovtool.cli`).

## Quick Start

```bash
# 1. List available inference devices
ovtool devices

# 2. Convert + INT4-quantize an LLM (downloaded from Hugging Face)
ovtool convert llm Qwen/Qwen3-0.6B -o ./qwen3-06b-int4

# 3. One-shot generation
ovtool generate -m ./qwen3-06b-int4 -d CPU "Describe OpenVINO in one sentence"

# 4. Interactive multi-turn chat
ovtool chat -m ./qwen3-06b-int4 -d GPU --opt perf_mode=LOW_LATENCY
```

## Subcommand Reference

### `ovtool models [remote|local] [query]`

**`remote`** (default) lists the built-in compatibility registry — HF models with
their verified (device × quantization × parameter) combinations — marking which
variants are available locally (`*`), without the convert command examples:

- asymmetric INT4 LLMs on NPU (symmetric INT4 required)
- diffusion or VLM pipelines on NPU (unsupported/hang in testing)
- image inputs for Qwen3-VL / Qwen3.5 on the current openvino-genai release
- `--max-new-tokens` exceeding the NPU static response budget (warning)

**`local`** lists every model found on disk with its path and size.

Local discovery scans `./models` plus **every root listed in the
`OVTOOL_MODELS_PATH` environment variable** (path-separator separated, e.g.
`OVTOOL_MODELS_PATH=D:\ovmodels;E:\more`), recursively: any directory
containing exported OpenVINO artifacts (`openvino_model.xml` /
`openvino_language_model.xml` / `model_index.json`) is auto-classified as
llm / vlm / image. The web UI catalog (`ovtool webui`) picks up the same
roots and additionally offers an **add-path box** in the sidebar: type any
directory, click `+`, and its models (found recursively) join the
category → model → quantization cascade for the session. Both the box and the
other UI preferences (dark/light theme, English/中文 language) live behind the
⚙ button in the top-right corner; enabling *Persist paths across restarts*
saves the whole panel — theme, language, extra roots and the UI state
(model/device selections and generation parameters, synced as you
change them) — to `~/.ovtool/webui_settings.json` so everything
survives restarts; disabling it removes the file and the next start
falls back to defaults.

### `ovtool devices`

Lists available OpenVINO devices with full device names, driver versions, and sub-devices (e.g. `GPU.0 / GPU.1`).

### `ovtool convert <kind> <model>`

| kind | Model family | Default export task |
|---|---|---|
| `llm` | Text LLMs | `text-generation-with-past` |
| `vlm` | Vision-language models | `image-text-to-text` |
| `image` | Diffusion image generation | Auto-selected per model (SD / SDXL / Flux / LCM) |
| `tts` | Text-to-speech (SpeechT5 / Kokoro) | `text-to-audio` (defaults to fp16 — small, quantization-sensitive models) |
| `embed` | Text embedding models (BGE / GTE / E5 / MiniLM ...) | `feature-extraction` (defaults to fp16) |
| `rerank` | Cross-encoder rerankers (BGE-Reranker / ms-marco ...) | `text-classification` (defaults to fp16) |

Common options:

- `-o DIR` output directory (default `./<model-name>-<quant-format>`)
- `--weight-format`: presets such as `fp32` / `fp16` / `int8` / `int4` / `int4_symg128` (symmetric, group 128)
- `--sym` / `--asym`: symmetric / asymmetric quantization (**symmetric INT4 is required for NPU**; asymmetric gives better accuracy on CPU/GPU)
- `--ratio 0.8`, `--group-size 64`: INT4 compression ratio and group size
- `--awq --dataset wikitext2`: activation-aware weight quantization (AWQ)
- `--trust-remote-code`: allow custom modeling code from the HF repo

Examples:

```bash
ovtool convert vlm openbmb/MiniCPM-V-2_6 -m ./minicpmv-int4 --sym
ovtool convert image stabilityai/sd-turbo -m ./sd-turbo-ir --weight-format int8
ovtool convert tts microsoft/speecht5_tts -o ./speecht5-fp16
ovtool convert tts hexgrad/Kokoro-82M --trust-remote-code -o ./kokoro
```

TTS notes: the SpeechT5 export automatically pulls the `speecht5_hifigan`
vocoder (override with `--vocoder`); Kokoro needs `pip install kokoro` at
export time (it also fetches misaki G2P data, incl. a spacy model, from the
network) plus `--trust-remote-code`, and copies its 54 `voices/*.bin` packs
into the output directory.

### `ovtool download`

Downloads a Hugging Face repo — or a single subfolder of it — to a local
directory, with a choice of **endpoint**: `huggingface.co` (default) or
`hf-mirror.com` (mirror for blocked networks), plus an optional `--proxy`:

```bash
# a whole repo
ovtool download Qwen/Qwen3-0.6B -o ./qwen3

# one quantization variant out of a multi-model repo
ovtool download dunegym/openvino-models \
    --subfolder llm/Qwen3-0.6B/int4-sym-g128 -o ./qwen3-sym

# via the mirror endpoint
ovtool download Qwen/Qwen3-0.6B --endpoint hf-mirror.com -o ./qwen3
```

Downloads resume per file (huggingface_hub cache) and print per-file
progress. The web UI offers the same feature in its settings panel: repo id,
optional subfolder, destination, endpoint and proxy, with a live progress
line — and the downloaded directory is added to the model catalog
automatically when it contains OpenVINO models.

### `ovtool generate` / `ovtool chat` (LLM)

Shared options: `-m` model directory; `-d` device (`CPU`/`GPU`/`NPU`/`AUTO`/`HETERO:GPU,CPU`…);
`--opt KEY=VALUE` runtime options (repeatable), e.g.:

- `--opt perf_mode=THROUGHPUT|LOW_LATENCY|CUMULATIVE_THROUGHPUT`
- `--opt inference_num_threads=8`
- `--opt num_streams=auto`

Generation parameters: `--max-new-tokens`, `--temperature` (>0 enables sampling), `--top-p`, `--top-k`,
`--repetition-penalty`, `--rng-seed`, `--stop-tokens`; `--no-stream` disables streaming; `--stats` prints TTFT/TPOT/throughput.

**NPU-specific options** (defaults applied automatically with `-d NPU`): `--max-prompt-len` (default 16384) and
`--min-response-len` (default 256) set the static-shape compile budget. NPU models must be converted with
`ovtool convert llm ... --weight-format int4 --sym` (symmetric quantization).

Chat-mode built-in commands: `/exit` to quit, `/reset` to clear history, `/system <text>` to set the system prompt.

### `ovtool serve` (OpenAI-compatible API)

Serves a converted LLM behind an OpenAI-format HTTP API, so any OpenAI client
library / tool can use it directly:

```bash
ovtool serve -m ./qwen3-06b-int4 -d GPU --port 8000
```

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | `messages` / `stream` (SSE) / `temperature` / `top_p` / `top_k` / `max_tokens` (also `max_completion_tokens`) / `stop` / `seed` / `n` (≤8) / `stream_options.include_usage` |
| `POST /v1/completions` | legacy text completions, same sampling params |
| `GET /v1/models`, `GET /health` | served-model listing / liveness |

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="ovtool")
resp = client.chat.completions.create(
    model="any", messages=[{"role": "user", "content": "hello"}])
```

Implementation notes: the chat template is rendered per request via the
model's own tokenizer (`apply_chat_template`), so the server is stateless and
safe across clients; a single pipeline is shared and generation is serialized
with a lock. `usage` token counts come from GenAI perf metrics. Optional
`--api-key KEY` requires `Authorization: Bearer KEY` on every request.
Tools/function calling, `logprobs` and image content are rejected with a 400
(use `ovtool vlm` for multimodal).

### `ovtool webui` (Browser UI)

```bash
ovtool webui --port 7860 --models-dir ./models
```

Opens a single-page UI (no build step, no CDN dependencies) over the models
directory (`<kind>/<model>/<variant>/` layout, same as the
[openvino-models](https://huggingface.co/dunegym/openvino-models) repo):

- **Chat tab** — pick an llm/vlm variant + device, Load, then multi-turn chat
  with SSE streaming, sampling params and token usage; VLM models can take
  image attachments (subject to the same GenAI image-input limitation)
- **Image tab** — pick a diffusion variant, generate with prompt / size /
  steps / guidance / seed controls; segmented loading (`TE,DENOISE,VAE`, e.g.
  `NPU,NPU,GPU`) fixes static geometry at load time, mirroring `--devices`
- **Devices tab** — live OpenVINO device table

One pipeline is resident at a time (loading runs in a background thread, the
UI polls status); switching models unloads the previous one. Every load goes
through the compatibility registry, so known-bad combos (e.g. asymmetric INT4
on NPU) fail fast with the registry's guidance. Backend is stdlib-only,
reusing the serve implementation for chat templating.

### `ovtool vlm` (Multimodal)

```bash
ovtool vlm -m ./minicpmv-int4 -d GPU -i ./photo.jpg "Describe the contents of this image"
```

`-i` may be repeated to pass multiple images; generation parameters are the same as for LLMs.

### `ovtool tts` (Text-to-Speech)

```bash
# SpeechT5: English TTS with the built-in default speaker (cmu-arctic x-vector)
ovtool tts -m ./speecht5-fp16 -d CPU "Describe OpenVINO in one sentence" --out ./out.wav

# Kokoro: pick a voice from the model's voices/ pack (54 available)
ovtool tts -m ./kokoro --speaker af_heart --language en-us --speed 1.0 "Hello!"
```

Options: `--speaker <voice>` (Kokoro voice name, e.g. `af_heart` / `am_michael`;
default: first pack, all names listed on first run), `--speaker-embedding FILE`
(explicit `.bin` override — for SpeechT5 a 512-float x-vector), `--language`
(Kokoro G2P: `en-us` / `en-gb` end-to-end; `es` / `fr-fr` / `hi` / `it` / `pt-br`
require espeak-ng), `--speed` (Kokoro), `--minlenratio` / `--maxlenratio` /
`--threshold` (SpeechT5 stop-behavior), `--stats`, `--out FILE` / `--out-dir`.
Backend-specific parameters passed to the wrong backend are ignored with a
warning. Output is 16-bit PCM mono WAV at the model's native rate
(SpeechT5 16 kHz, Kokoro 24 kHz). **CPU is considerably faster than GPU here**
(verified ~10x: these are small autoregressive models where per-step iGPU
overhead dominates). NPU is rejected by the registry (not supported upstream).

### `ovtool embed` / `ovtool rerank` (Retrieval)

```bash
# convert (pooling config from sentence-transformers is preserved automatically)
ovtool convert embed BAAI/bge-small-en-v1.5 -o ./bge-small
ovtool convert rerank BAAI/bge-reranker-v2-m3 -o ./bge-reranker

# vectorize documents (prints dim / norm / preview; --json for full vectors)
ovtool embed -m ./bge-small "OpenVINO is a toolkit" "A cat video"

# retrieval ranking: cosine similarity of the query against every document
ovtool embed -m ./bge-small --query "what is OpenVINO?" \
    --query-instruction "Represent this sentence for searching relevant passages: " \
    "OpenVINO optimizes inference" "A cat video" "An inference toolkit from Intel"

# cross-encoder reranking (sigmoid relevance scores, sorted)
ovtool rerank -m ./bge-reranker "what is OpenVINO?" \
    "OpenVINO is an open-source inference toolkit" "A cat sits on the mat" \
    "OpenVINO 支持 CPU、GPU 和 NPU 推理加速"
```

`embed` options: `--query` (query-side embedding; with document texts it
becomes cosine ranking), `--pooling cls|mean|last_token` (default: auto from
the copied `1_Pooling/config.json`, else mean — GenAI does not auto-detect
pooling), `--no-normalize`, `--query-instruction` / `--embed-instruction`
(bge / e5 style prefixes), `--max-length`, `--batch-size`, `--json [--out FILE]`.
`rerank` options: `--top-n`, `--instruction` (Qwen3 reranking task),
`--json`. Both take `-d device` and `--opt`. BGE models work best with the
query instruction above; E5 models use `--query-instruction "query: "` /
`--embed-instruction "passage: "`.

**Qwen3 retrieval models**: `Qwen3-Embedding` converts via `convert embed`
(feature-extraction) — `ovtool embed` auto-applies its LAST_TOKEN pooling and
the official query instruction. `Qwen3-Reranker` converts via `convert llm`
(text-generation-with-past — the LLM export path, not `convert rerank`);
`ovtool rerank` detects the qwen3 model type and wraps query/documents in the
official yes/no instruction template automatically (GenAI feeds them to the
model verbatim, so without the template the scores are near-random).
`--instruction` customizes the reranking task text.

### `ovtool image` / `ovtool image2image` (Diffusion)

```bash
ovtool image -m ./sd-turbo-ir -d GPU "a corgi surfing a wave" \
    --width 512 --height 512 --steps 8 --guidance-scale 1.0 --seed 42 \
    --out-dir ./generated
```

Options: `--width/--height`, `--steps` (denoising steps), `--guidance-scale`, `--num-images`,
`--negative-prompt`, `--seed`, `--scheduler` (e.g. `LCM`, `EULER_ANCESTRAL`, depending on the model), `--out-dir`.

**Segmented multi-device execution** (`--devices TEXT,DENOISE,VAE`): each pipeline component
gets its own device, which is how diffusion runs on the NPU (VAE decode is not NPU-capable
and stays on GPU/CPU). Shapes are fixed statically to `--width/--height/--num-images`:

```bash
ovtool image -m ./sd-turbo-int8 -d GPU --devices NPU,NPU,GPU \
    "a corgi surfing a wave" --steps 4 --guidance-scale 1.0 --seed 42
```

Verified with sd-turbo int8 (512px, 4 steps): first run compiles the NPU graphs (~2 min,
cached under `<model>/cache`), subsequent runs ~12s — on par with an all-GPU run (~14s) at
near-identical output for the same seed. The registry blocks whole-pipeline `-d NPU` for
image models but allows the segmented form; putting the VAE on NPU triggers a warning.

> Note: diffusion models are officially recommended to run on **GPU** (which is also the default device).

## Device Selection Guide

| Device | Best for | Notes |
|---|---|---|
| CPU | General use, accelerated by AVX2/AVX-512/AMX | LLMs work with INT4/INT8 |
| GPU (iGPU / Arc / DC GPU) | Best for diffusion; good LLM throughput | Requires Intel graphics drivers |
| NPU (Core Ultra) | Low-power LLM inference; **text-mode VLM inference (Qwen3-VL int4-sym, verified)**; diffusion in segmented mode (`image --devices NPU,NPU,GPU`) | **Symmetric INT4 required for LLMs** (`convert ... --sym`); static-shape execution — LLM budget via `--max-prompt-len` (default 16384) / `--min-response-len` (default 256), measured ~21 tok/s on Qwen3-0.6B with the local NPU 3720 |
| AUTO / HETERO | Automatic selection / mixed execution | Useful when device capabilities are uncertain |

## Code Structure

```
ovtool/
├── cli.py        # Entry point & subcommand registration
├── devices.py    # Device enumeration / validation
├── convert.py    # optimum-intel export + weight quantization
├── llm.py        # LLMPipeline: generate / chat
├── server.py     # OpenAI-compatible API server (serve)
├── webui.py/.html# Browser UI (webui)
├── vlm.py        # VLMPipeline: image-text multimodal
├── tts.py        # Text2SpeechPipeline: SpeechT5 / Kokoro speech synthesis
├── embed.py      # TextEmbeddingPipeline / TextRerankPipeline: retrieval
└── imagegen.py   # Text2Image / Image2Image
```

## Verified (local machine: Core Ultra 5 125H + Arc Pro iGPU + NPU 3720)

- `devices` / `--help` for all subcommands: ✅
- LLM conversion + INT4 quantization (**Qwen3-0.6B**, reasoning model): ✅ ~55 tok/s on GPU, multi-turn chat OK on CPU, math comparison answered correctly
- `generate` one-shot generation (CPU / GPU, streaming & non-streaming, `--stats`, `--opt perf_mode=...`): ✅ (~50 tok/s @0.5B-int4 on iGPU)
- `chat` multi-turn interaction (incl. `/exit` `/reset` `/system`): ✅
- Diffusion conversion + INT8 quantization (sd-turbo): ✅
- Text-to-image model expansion (LCM-Dreamshaper-v7 / SD-1.5 / SSD-1B, int8 + int4-g64 each): ✅ generation verified on iGPU (LCM 8-step 512px ~19s; SSD-1B 25-step 1024px ~89s); includes the SDXL dual-tokenizer export fix (`tokenizer_2` IR)
- **SD3.5-medium** (gated repo, 2.5B MMDiT + T5-XXL): ✅ both variants generation-verified on iGPU (1024px/28 steps: int8 ~3m22s, int4-g64 ~3m36s); required the SD3 pipeline-class fix and generalized triple-tokenizer export (`tokenizer_2` + `tokenizer_3` IR)
- **FLUX.2-klein-4B** (Apache-2.0, 3.9B distilled, Qwen3 text encoder): ✅ both variants generation-verified on iGPU (1024px/8 steps: int8 ~2m06s, int4-g64 ~2m13s) after upgrading to optimum-intel 2.2.0
- `image` text-to-image / `image2image` (GPU, seed reproducibility): ✅ (512×512×4 steps in seconds)
- **NPU inference (Qwen3-0.6B symmetric INT4)**: ✅ TTFT ~1.5s, ~21 tok/s; `--max-prompt-len` / `--min-response-len` static-shape options verified
- **Qwen3.5-0.8B / Qwen3.5-2B** (new native multimodal `qwen3_5` architecture, sym/asym INT4): ✅ text generation OK on GPU (2B ~33 tok/s; 0.8B NPU compile extremely slow)
- **Qwen3-VL-2B-Instruct INT4**: ✅ text generation OK
- **VLM on NPU** (text mode): ✅ Qwen3-VL-2B / Qwen3-VL-4B int4-sym-g128 — correct output on NPU 3720, first compile 34s/94s (blob-cached in `<model>/cache`), ~5-11s per 40-token answer; qwen3_5 (silent process kill) and gemma4 (garbage output, CPU-fine) stay blocked
- **TTS — SpeechT5** (`microsoft/speecht5_tts`, fp16 encoder/decoder/postnet/vocoder + tokenizer IR): ✅ CPU 4.1s of 16 kHz speech in 2.3s (~28.8k samples/s); GPU works but ~10x slower; default speaker embedding built into GenAI
- **TTS — Kokoro-82M** (fp16 single IR + 54 voice packs): ✅ CPU 4.7s of 24 kHz speech in 2.7s with `--speaker af_heart --language en-us`; default-voice fallback, bad-voice listing and wrong-backend param warnings verified
- **Embeddings — bge-small-en-v1.5** (fp16, 384-dim): ✅ CLS pooling auto-applied from the copied `1_Pooling` config; cosine retrieval ranking semantically correct on CPU/GPU; JSON vector dump verified
- **Rerank — bge-reranker-v2-m3** (fp16, 568M multilingual): ✅ sigmoid scores rank a relevant English doc 0.9999 / Chinese doc 0.80 / irrelevant 0.0000 for an English query; `--top-n` verified on CPU
- **Qwen3-Embedding-0.6B** (fp16 via `convert embed`, 1024-dim): ✅ LAST_TOKEN pooling + official query instruction auto-applied; cosine ranking correct across en+zh (relevant 0.81–0.83 vs irrelevant 0.23)
- **Qwen3-Reranker-0.6B** (int4 via `convert llm`): ✅ official yes/no template auto-applied — P(yes) 0.986/0.969 for relevant en/zh docs vs 0.010 irrelevant (raw query+doc without the template scores near-random; never bypass it)
- **google/gemma-4-E2B-it** (VLM, effective-2B MatFormer with per-layer embeddings, full five-variant ladder): ✅ text (en+zh) and **image input** verified on int4-asym-g128 — ~26 tok/s CPU, ~20 tok/s GPU with 0.5s TTFT; the repo's own `chat_template.jinja` (tool-calling capable) renders fine in GenAI
- The `vlm` multimodal path is implemented per the official openvino-genai API; image+text inference was not verified end-to-end (see known limitations)

## Known Limitations (measured 2026-09)

1. **Image input for Qwen3-VL / Qwen3.5**: conversion succeeds, but image+text inference via `VLMPipeline` fails with `Argument shapes are inconsistent` on GenAI 2026.3.1 (reproduced at every resolution). The master branch already has dedicated `InputsEmbedderQwen3VL/Qwen3_5` implementations — waiting for the next release. **Text-only mode is unaffected.**
2. **VLM on NPU** (re-measured 2026-09): GenAI 2026.3.1 has a real NPU VLM path (static-KV language model + `MAX_PROMPT_LEN` budget, embedder auto-fallback to CPU; chat mode replays the full history). Verified: **Qwen3-VL-2B/4B int4-sym text mode works** (`ovtool vlm -d NPU`, compile 34s/94s then cached). Two families remain broken and stay registry-blocked: **qwen3_5** (0.8B and 2B both kill the process silently during NPU compile — no Python traceback, driver recovers afterwards) and **gemma4** (compiles and generates but output is garbage on NPU while the same sym weights are correct on CPU — NPU numerics issue). Image input is blocked by the upstream GenAI shape bug on ALL devices (see #1), not NPU-specifically.
3. **FLUX.2-klein export**: fixed by optimum-intel 2.2.0 (FLUX.2 support PR #1809 + dynamic-sequence fix #1846; the old 2.1.0 `pos_embed` tracing failure is gone). FLUX.2-klein-4B converts and generates on iGPU (1024px/8 steps ~2m06s int8 / ~2m13s int4-g64). Note the conversion env pairing: optimum-intel 2.2.0 + diffusers 0.39 (0.40 pulls an LTX2→Gemma4Unified import chain that needs unreleased transformers) + transformers 5.5.4; transformers 5.3+ drops qwen3_5 re-conversion (`pip install transformers==5.2.0` to restore — already-converted models are unaffected).
4. **optimum version guards**: optimum-intel 2.1.0 pins stale `MAX_TRANSFORMERS_VERSION` values on newer architectures such as qwen3-vl/qwen2-vl; `convert vlm` relaxes them automatically (`_relax_stale_version_guards`). qwen3_5 additionally requires transformers==5.2.x (5.3+ removed `Qwen3_5DynamicCache`, while optimum pins `<5.6`; 5.2 satisfies both).
5. **TTS scope of the current GenAI release** (2026.3.1): only SpeechT5 and Kokoro-82M are supported by `Text2SpeechPipeline` — **Qwen3-TTS is not** (community OpenVINO conversions exist on HF but do not run through GenAI). Kokoro zh/ja voices ship in the pack but are not supported end-to-end (G2P); non-English languages (es/fr-fr/hi/it/pt-br) require espeak-ng installed.
6. **Gemma-4 base models ship no chat template**: `google/gemma-4-E2B` (base) carries none in any file and transformers has no Gemma4 default — GenAI chat/VLM inference then fails with `chat_template.empty()`. The `-it` sibling repo has `chat_template.jinja`; `convert vlm` warns when the export lacks a template. The base model is also not instruction-aligned (repeats itself, blank image descriptions) and is therefore not shipped in the models repo — use `-it`.

## Implementation Notes (lessons learned)

- **Diffusion tokenizer location**: GenAI's SD pipelines look for `openvino_tokenizer.xml` inside the `tokenizer/` component subfolder, following the `text_encoder → tokenizer` path convention. `ovtool convert image` places it there automatically (LLM/VLM keep it at the model root).
- **openvino_tokenizers extension**: inference modules pre-`import openvino_tokenizers` to register the custom-op extension, preventing Tokenizer load failures.
- **CLIP-style slow tokenizers** require `sentencepiece` / `tiktoken` to convert (added as dependencies).
- The `rng_seed` generation parameter and the `ov.Tensor(N,H,W,C)` image result are adapted to the openvino-genai 2026.3 API.
- optimum-intel 2.1.0 does not save the converted tokenizer with `save_pretrained`, so `convert` converts and saves it via openvino-tokenizers itself.
- **optimum-intel drops `model_kwargs` on the Python export path** (`from_pretrained(export=True)` → `_export` → `main_export` forwards no kwargs), which the SpeechT5 exporter requires for the vocoder — `convert tts` therefore calls `optimum.exporters.openvino.main_export` directly, with the library inferred per model (`transformers` vs optimum-intel's `kokoro` detection, without which the model_type-less Kokoro config crashes `AutoConfig`).
- **In-place fp16 re-serialization fails on Windows**: `core.read_model()` keeps the original `.bin` memory-mapped, so `ov.serialize` cannot reopen the same path — serialize to a sibling `.fp16.*` file, release the model, then `os.replace` over the original.
- **GPU TTS results are remote tensors**: `Tensor.data` raises `Not Implemented` on GPU outputs; `tts.py` copies to a host `ov.Tensor` via `copy_to` first (CPU tensors read directly).
- **GenAI does not auto-detect embedding pooling**: `TextEmbeddingPipeline` defaults to CLS regardless of the model. `convert embed` preserves the sentence-transformers `1_Pooling/config.json` next to the IR and `ovtool embed` applies it (bge = CLS, MiniLM = mean), with `--pooling` as override. Rerank scores are post-processed by GenAI itself (sigmoid for single-logit cross-encoders).
- **Broken `tokenizer.json` serialization for tiktoken-backed tokenizers**: under transformers 5.x, `save_pretrained` on tokenizers of newer models such as Qwen3 writes a `tokenizer.json` that encodes to empty results via the tokenizers library (silently broken). `_save_tokenizer` now loads the tokenizer from the original HF repo first and probes each candidate with a non-empty encoding check.
