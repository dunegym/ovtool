"""Text-to-speech generation via openvino_genai.Text2SpeechPipeline.

Supports the two backends of the current openvino-genai release:

- SpeechT5 (microsoft/speecht5_tts + speecht5_hifigan vocoder): a default
  speaker embedding (the 7306-th cmu-arctic x-vector) is compiled into
  openvino-genai, so no voice file is needed; --speaker-embedding overrides it.
- Kokoro (hexgrad/Kokoro-82M): a speaker embedding from the model's
  voices/*.bin pack must be passed explicitly (--speaker <name>).

Audio is written as 16-bit PCM WAV (stdlib wave module, no extra dependency).
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

import openvino_tokenizers  # noqa: F401  (registers custom-op extension)

KOKORO_LANGS = ("en-us", "en-gb", "es", "fr-fr", "hi", "it", "pt-br")


def detect_backend(model_dir: str) -> str:
    """'kokoro' | 'speecht5' from the exported artifacts."""
    d = Path(model_dir)
    if (d / "voices").is_dir():
        return "kokoro"
    try:
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if str(cfg.get("model_type", "")).lower() == "kokoro":
            return "kokoro"
    except (OSError, ValueError):
        pass
    return "speecht5"


def load_speaker_embedding(path: Path, shape) -> "object":
    """float32 .bin file -> ov.Tensor reshaped to the model's expected shape."""
    import openvino as ov

    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        raise SystemExit(f"speaker embedding file is empty: {path}")
    try:
        return ov.Tensor(data.reshape([int(d) for d in shape]))
    except ValueError as e:
        raise SystemExit(
            f"speaker embedding {path.name} ({data.size} floats) does not match "
            f"the model's expected shape {list(shape)}: {e}") from e


def resolve_speaker(pipe, model_dir: str, args: argparse.Namespace):
    """--speaker-embedding file > --speaker voice name > first voice (kokoro)
    > None (SpeechT5 built-in default)."""
    if args.speaker_embedding:
        return load_speaker_embedding(Path(args.speaker_embedding),
                                      pipe.get_speaker_embedding_shape())

    voices_dir = Path(model_dir) / "voices"
    bins = sorted(voices_dir.glob("*.bin")) if voices_dir.is_dir() else []
    if args.speaker:
        if not bins:
            raise SystemExit(f"no voices/ directory with .bin packs in {model_dir}")
        match = [b for b in bins if b.stem.lower() == args.speaker.lower()]
        if not match:
            names = ", ".join(b.stem for b in bins)
            raise SystemExit(f"unknown voice '{args.speaker}'; available: {names}")
        return load_speaker_embedding(match[0], pipe.get_speaker_embedding_shape())
    if bins:
        # Kokoro requires an explicit embedding; default to the first pack
        print(f"[ovtool] using voice '{bins[0].stem}' "
              f"(--speaker to change; available: {', '.join(b.stem for b in bins)})")
        return load_speaker_embedding(bins[0], pipe.get_speaker_embedding_shape())
    # SpeechT5: GenAI falls back to its compiled-in default x-vector
    return None


def _speech_props(args: argparse.Namespace, backend: str) -> dict:
    """Backend-specific generation properties; mismatched params are dropped
    with a warning instead of reaching the runtime."""
    kokoro = {"language": args.language, "speed": args.speed}
    speecht5 = {"minlenratio": args.minlenratio, "maxlenratio": args.maxlenratio,
                "threshold": args.threshold}
    props = {}
    for name, v in (kokoro if backend == "kokoro" else speecht5).items():
        if v is not None and not (name == "speed" and v == 1.0):
            props[name] = v
    dropped = (speecht5 if backend == "kokoro" else kokoro)
    for name, v in dropped.items():
        if v is not None and not (name == "speed" and v == 1.0):
            print(f"[ovtool] warning: --{name.replace('_', '-')} applies to "
                  f"{'SpeechT5' if backend == 'kokoro' else 'Kokoro'} models; ignoring")
    if backend == "kokoro" and args.language not in (None,) + KOKORO_LANGS:
        print(f"[ovtool] warning: language '{args.language}' is not in the "
              f"end-to-end supported set {KOKORO_LANGS} (zh/ja need espeak-ng and "
              "are not yet supported end-to-end)")
    return props


def _waveform(tensor) -> np.ndarray:
    """Speech ov.Tensor -> flat float32 numpy array. GPU devices hand back a
    remote tensor whose .data accessor is not implemented; copy it to a host
    tensor first."""
    import openvino as ov

    try:
        return np.asarray(tensor.data).reshape(-1).astype(np.float32)
    except RuntimeError:
        host = ov.Tensor(tensor.element_type, tensor.shape)
        tensor.copy_to(host)
        return np.asarray(host.data).reshape(-1).astype(np.float32)


def save_wav(result, out: Path) -> tuple[str, float]:
    """First generated speech -> 16-bit PCM mono WAV at `out`; returns
    (path, seconds of audio)."""
    data = _waveform(result.speeches[0])
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(result.output_sample_rate))
        w.writeframes(pcm.tobytes())
    return str(out), data.size / result.output_sample_rate


def run_tts(args: argparse.Namespace) -> None:
    import openvino_genai as ovgenai

    from .devices import resolve_device
    from .llm import compile_options

    device = resolve_device(args.device)
    backend = detect_backend(args.model)
    print(f"[ovtool] loading TTS ({backend}) {args.model} on {device} ...")
    opts = compile_options(args)
    pipe = ovgenai.Text2SpeechPipeline(
        args.model, device, **({"config": opts} if opts else {}))

    speaker = resolve_speaker(pipe, args.model, args)
    props = _speech_props(args, backend)

    import time
    t0 = time.time()
    result = pipe.generate(args.text, speaker, **props)
    elapsed = time.time() - t0

    out = Path(args.out) if args.out else \
        Path(args.out_dir) / f"tts_{time.strftime('%H%M%S')}.wav"
    path, duration = save_wav(result, out)
    print(f"[ovtool] {duration:.1f}s of speech @ {result.output_sample_rate} Hz "
          f"in {elapsed:.1f}s -> {path}")

    if args.stats:
        pm = result.perf_metrics
        try:
            if pm.m_evaluated:
                print(f"[stats] throughput {pm.throughput.mean:.1f} samples/s | "
                      f"generation {pm.generate_duration.mean / 1000:.1f}s")
        except AttributeError:
            print("[stats] perf metrics not available on this GenAI release")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("tts", help="Text-to-speech generation (SpeechT5 / Kokoro)",
                       description="Generate speech from text with a converted TTS model.\n"
                                   "Examples:\n"
                                   "  ovtool convert tts microsoft/speecht5_tts -o ./speecht5-fp16\n"
                                   "  ovtool tts -m ./speecht5-fp16 -d CPU \"Hello from ovtool\"\n"
                                   "  ovtool convert tts hexgrad/Kokoro-82M --trust-remote-code -o ./kokoro\n"
                                   "  ovtool tts -m ./kokoro --speaker af_heart --language en-us \"Hello!\"",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True,
                   help="Model directory (OpenVINO IR from 'convert tts')")
    p.add_argument("-d", "--device", default="CPU",
                   help="Inference device (default CPU; TTS on NPU is not supported)")
    p.add_argument("text", help="Text to synthesize")
    p.add_argument("--speaker", default=None, metavar="VOICE",
                   help="Kokoro voice name, resolved to <model>/voices/<VOICE>.bin "
                        "(default: first voice pack)")
    p.add_argument("--speaker-embedding", default=None, metavar="FILE",
                   help="Explicit speaker-embedding .bin file (overrides --speaker; "
                        "SpeechT5: 512-float x-vector, otherwise the built-in default)")
    p.add_argument("--language", default=None,
                   help="Kokoro G2P language: en-us / en-gb / es / fr-fr / hi / it / pt-br")
    p.add_argument("--speed", type=float, default=1.0, help="Kokoro speech speed multiplier")
    p.add_argument("--minlenratio", type=float, default=None, help="SpeechT5 min length ratio")
    p.add_argument("--maxlenratio", type=float, default=None, help="SpeechT5 max length ratio")
    p.add_argument("--threshold", type=float, default=None, help="SpeechT5 stop threshold")
    p.add_argument("--opt", action="append", metavar="KEY=VALUE",
                   help="Runtime option, e.g. --opt perf_mode=THROUGHPUT (repeatable)")
    p.add_argument("--stats", action="store_true", help="Print performance stats")
    p.add_argument("--out", default=None, help="Output WAV path (default ./generated-tts/tts_<time>.wav)")
    p.add_argument("--out-dir", default="./generated-tts", help="Output directory when --out is not given")
    p.set_defaults(func=run_tts)
