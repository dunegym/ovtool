# ovtool — A Comprehensive OpenVINO Command-Line Tool

A multi-purpose inference CLI built on [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai), supporting:

- **LLM inference**: one-shot generation + interactive multi-turn chat, streaming output, full sampling controls
- **OpenAI-compatible API server**: expose a converted LLM behind `POST /v1/chat/completions` (incl. SSE streaming), `POST /v1/completions` and `GET /v1/models`
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
ovtool convert llm Qwen/Qwen2.5-0.5B-Instruct -m ./qwen05-int4

# 3. One-shot generation
ovtool generate -m ./qwen05-int4 -d CPU "Describe OpenVINO in one sentence"

# 4. Interactive multi-turn chat
ovtool chat -m ./qwen05-int4 -d GPU --opt perf_mode=LOW_LATENCY
```

## Subcommand Reference

### `ovtool models`

Lists the built-in compatibility registry: HF models with their verified
(device × quantization × parameter) combinations and the matching `convert`
commands. All inference subcommands consult this registry before loading a
model and refuse known-bad configurations, e.g.:

- asymmetric INT4 LLMs on NPU (symmetric INT4 required)
- diffusion or VLM pipelines on NPU (unsupported/hang in testing)
- image inputs for Qwen3-VL / Qwen3.5 on the current openvino-genai release
- `--max-new-tokens` exceeding the NPU static response budget (warning)

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

> Note: diffusion models are officially recommended to run on **GPU** (which is also the default device).
> On NPU, diffusion runs in a segmented fashion (text encoder + UNet on NPU, VAE decoder on GPU); this tool does not orchestrate that mode automatically yet.

## Device Selection Guide

| Device | Best for | Notes |
|---|---|---|
| CPU | General use, accelerated by AVX2/AVX-512/AMX | LLMs work with INT4/INT8 |
| GPU (iGPU / Arc / DC GPU) | Best for diffusion; good LLM throughput | Requires Intel graphics drivers |
| NPU (Core Ultra) | Low-power LLM inference | **Symmetric INT4 required (`convert ... --sym`)**; static-shape execution — set the compile-time budget via `--max-prompt-len` (default 1024) / `--min-response-len` (default 128); measured ~21 tok/s on Qwen3-0.6B with the local NPU 3720 |
| AUTO / HETERO | Automatic selection / mixed execution | Useful when device capabilities are uncertain |

## Code Structure

```
ovtool/
├── cli.py        # Entry point & subcommand registration
├── devices.py    # Device enumeration / validation
├── convert.py    # optimum-intel export + weight quantization
├── llm.py        # LLMPipeline: generate / chat
├── server.py     # OpenAI-compatible API server (serve)
├── vlm.py        # VLMPipeline: image-text multimodal
└── imagegen.py   # Text2Image / Image2Image
```

## Verified (local machine: Core Ultra 5 125H + Arc Pro iGPU + NPU 3720)

- `devices` / `--help` for all subcommands: ✅
- LLM conversion + INT4 quantization (Qwen2.5-0.5B-Instruct, 322MB int4 IR): ✅
- LLM conversion + INT4 quantization (**Qwen3-0.6B**, reasoning model): ✅ ~55 tok/s on GPU, multi-turn chat OK on CPU, math comparison answered correctly
- `generate` one-shot generation (CPU / GPU, streaming & non-streaming, `--stats`, `--opt perf_mode=...`): ✅ (~50 tok/s @0.5B-int4 on iGPU)
- `chat` multi-turn interaction (incl. `/exit` `/reset` `/system`): ✅
- Diffusion conversion + INT8 quantization (sd-turbo): ✅
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
