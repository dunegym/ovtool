# ovtool — A Comprehensive OpenVINO Command-Line Tool

A multi-purpose inference CLI built on [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai), supporting:

- **LLM inference**: one-shot generation + interactive multi-turn chat, streaming output, full sampling controls
- **OpenAI-compatible API server**: expose a converted LLM behind `POST /v1/chat/completions` (incl. SSE streaming), `POST /v1/completions` and `GET /v1/models`
- **Browser WebUI**: pick any converted model, load it on a device and chat / generate images from the browser (`ovtool webui`)
- **Multimodal (VLM) inference**: image + text Q&A (converted LLaVA / Qwen-VL / MiniCPM-V / InternVL models)
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
```

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

**NPU-specific options** (defaults applied automatically with `-d NPU`): `--max-prompt-len` (default 1024) and
`--min-response-len` (default 128) set the static-shape compile budget. NPU models must be converted with
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
| NPU (Core Ultra) | Low-power LLM inference; diffusion in segmented mode (`image --devices NPU,NPU,GPU`) | **Symmetric INT4 required for LLMs** (`convert ... --sym`); static-shape execution — LLM budget via `--max-prompt-len` (default 1024) / `--min-response-len` (default 128), measured ~21 tok/s on Qwen3-0.6B with the local NPU 3720 |
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
- `image` text-to-image / `image2image` (GPU, seed reproducibility): ✅ (512×512×4 steps in seconds)
- **NPU inference (Qwen3-0.6B symmetric INT4)**: ✅ TTFT ~1.5s, ~21 tok/s; `--max-prompt-len` / `--min-response-len` static-shape options verified
- **Qwen3.5-0.8B / Qwen3.5-2B** (new native multimodal `qwen3_5` architecture, sym/asym INT4): ✅ text generation OK on GPU (2B ~33 tok/s; 0.8B NPU compile extremely slow)
- **Qwen3-VL-2B-Instruct INT4**: ✅ text generation OK
- The `vlm` multimodal path is implemented per the official openvino-genai API; image+text inference was not verified end-to-end (see known limitations)

## Known Limitations (measured 2026-09)

1. **Image input for Qwen3-VL / Qwen3.5**: conversion succeeds, but image+text inference via `VLMPipeline` fails with `Argument shapes are inconsistent` on GenAI 2026.3.1 (reproduced at every resolution). The master branch already has dedicated `InputsEmbedderQwen3VL/Qwen3_5` implementations — waiting for the next release. **Text-only mode is unaffected.**
2. **VLM on NPU**: text mode of Qwen3.5-0.8B triggered `ZE_RESULT_ERROR_DEVICE_LOST` (driver hang, requires process restart) after a long compile on NPU. Recommend running only symmetric-INT4 pure LLMs on NPU (verified with qwen3-0.6b-sym).
3. **FLUX.2-klein export**: optimum-intel 2.1.0 fails while tracing `pos_embed` with `Axis out of rank range` (tracked upstream in [issue #1767](https://github.com/huggingface/optimum-intel/issues/1767)). For image generation use SD/SDXL/Flux.1-family models (sd-turbo verified).
4. **optimum version guards**: optimum-intel 2.1.0 pins stale `MAX_TRANSFORMERS_VERSION` values on newer architectures such as qwen3-vl/qwen2-vl; `convert vlm` relaxes them automatically (`_relax_stale_version_guards`). qwen3_5 additionally requires transformers==5.2.x (5.3+ removed `Qwen3_5DynamicCache`, while optimum pins `<5.6`; 5.2 satisfies both).

## Implementation Notes (lessons learned)

- **Diffusion tokenizer location**: GenAI's SD pipelines look for `openvino_tokenizer.xml` inside the `tokenizer/` component subfolder, following the `text_encoder → tokenizer` path convention. `ovtool convert image` places it there automatically (LLM/VLM keep it at the model root).
- **openvino_tokenizers extension**: inference modules pre-`import openvino_tokenizers` to register the custom-op extension, preventing Tokenizer load failures.
- **CLIP-style slow tokenizers** require `sentencepiece` / `tiktoken` to convert (added as dependencies).
- The `rng_seed` generation parameter and the `ov.Tensor(N,H,W,C)` image result are adapted to the openvino-genai 2026.3 API.
- optimum-intel 2.1.0 does not save the converted tokenizer with `save_pretrained`, so `convert` converts and saves it via openvino-tokenizers itself.
- **Broken `tokenizer.json` serialization for tiktoken-backed tokenizers**: under transformers 5.x, `save_pretrained` on tokenizers of newer models such as Qwen3 writes a `tokenizer.json` that encodes to empty results via the tokenizers library (silently broken). `_save_tokenizer` now loads the tokenizer from the original HF repo first and probes each candidate with a non-empty encoding check.
