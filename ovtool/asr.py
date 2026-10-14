"""Speech recognition (ASR) via openvino_genai.ASRPipeline / WhisperPipeline.

Supports Whisper-family exports ('convert asr'): audio is read from a WAV
file (any sample rate, mono or stereo; stdlib wave module), downmixed to
mono and linearly resampled to 16 kHz float32 — the format both pipelines
expect. The newer ASRPipeline (GenAI 2026.3+) is preferred, falling back to
WhisperPipeline on older releases.
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

# registers the openvino_tokenizers custom-op extension with the runtime
import openvino_tokenizers  # noqa: F401

TARGET_RATE = 16000


def read_wav_16k_mono(path: str) -> tuple[np.ndarray, int]:
    """WAV file -> (float32 mono samples in [-1, 1], original sample rate).

    16-bit PCM and 32-bit float WAVs are accepted; any channel count is
    averaged down to mono; rates other than 16 kHz are linearly resampled
    (adequate for mel-spectrogram front-ends like Whisper's)."""
    with wave.open(str(path), "rb") as w:
        nch, sw, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(frames, dtype=np.float32).copy()
    else:
        raise SystemExit(f"unsupported WAV sample width {sw * 8} bit "
                         f"(16-bit PCM or 32-bit float expected): {path}")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    if rate != TARGET_RATE:
        x = np.linspace(0.0, data.size / rate, num=data.size, endpoint=False)
        xi = np.linspace(0.0, data.size / rate,
                         num=int(data.size * TARGET_RATE / rate), endpoint=False)
        data = np.interp(xi, x, data).astype(np.float32)
    return np.clip(data, -1.0, 1.0), rate


def _open_pipeline(ovgenai, model: str, device: str, opts: dict):
    """ASRPipeline where available; WhisperPipeline on older GenAI."""
    cls = getattr(ovgenai, "ASRPipeline", None) or ovgenai.WhisperPipeline
    return cls(model, device, **({"config": opts} if opts else {}))


def run_asr(args: argparse.Namespace) -> None:
    import time

    import openvino_genai as ovgenai

    from .devices import resolve_device
    from .llm import compile_options

    device = resolve_device(args.device)
    audio, src_rate = read_wav_16k_mono(args.audio)
    print(f"[ovtool] transcribing {args.audio} "
          f"({audio.size / TARGET_RATE:.1f}s audio, {src_rate} Hz source) "
          f"on {device} ...")

    pipe = _open_pipeline(ovgenai, args.model, device, compile_options(args))
    gen_kwargs = {}
    if args.language:
        gen_kwargs["language"] = args.language
    if args.task:
        gen_kwargs["task"] = args.task
    if args.initial_prompt:
        gen_kwargs["initial_prompt"] = args.initial_prompt
    if args.timestamps:
        gen_kwargs["return_timestamps"] = True
    t0 = time.time()
    result = pipe.generate(audio.tolist(), **gen_kwargs)
    elapsed = time.time() - t0

    text = result.texts[0] if getattr(result, "texts", None) else str(result)
    print(text.strip())
    lang = getattr(result, "language", "")
    if lang:
        print(f"[ovtool] detected language: {lang}")
    chunks = getattr(result, "chunks", None)
    if args.timestamps and chunks:
        # chunks is one inner list per batch item; take the first
        flat = chunks[0] if isinstance(chunks[0], (list, tuple)) else chunks

        def part(chunk, name):
            v = getattr(chunk, name)
            return v() if callable(v) else v  # exposed as method or property

        for c in flat:
            print(f"  [{part(c, 'start_ts'):7.2f} -> {part(c, 'end_ts'):7.2f}] "
                  f"{part(c, 'text').strip()}")
    if args.stats:
        dur = audio.size / TARGET_RATE
        print(f"[stats] audio {dur:.1f}s transcribed in {elapsed:.1f}s "
              f"({dur / max(elapsed, 1e-9):.2f}x realtime)")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("asr", help="Speech recognition (Whisper)",
                       description="Transcribe a WAV file with a converted ASR model "
                                   "(Whisper family).\n"
                                   "Examples:\n"
                                   "  ovtool convert asr openai/whisper-small -o ./whisper-small-fp16\n"
                                   "  ovtool asr -m ./whisper-small-fp16 -d CPU recording.wav\n"
                                   "  ovtool asr -m ./whisper-small -d GPU zh.wav --language zh --timestamps\n"
                                   "  ovtool asr -m ./whisper-small -d CPU jp.wav --task translate  # to English",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True,
                   help="Model directory (OpenVINO IR from 'convert asr')")
    p.add_argument("-d", "--device", default="CPU",
                   help="Inference device (default CPU)")
    p.add_argument("audio", help="WAV file (any rate, mono or stereo; 16-bit PCM "
                                 "or 32-bit float)")
    p.add_argument("--language", default=None,
                   help="Force the spoken language (e.g. zh, en); default: detect")
    p.add_argument("--task", choices=["transcribe", "translate"], default=None,
                   help="translate renders the transcript into English "
                        "(multilingual models only)")
    p.add_argument("--initial-prompt", default=None,
                   help="Previous-transcript style/hotword hint for the first window")
    p.add_argument("--timestamps", action="store_true",
                   help="Print segment-level timestamps")
    p.add_argument("--opt", action="append", metavar="KEY=VALUE",
                   help="Runtime option, e.g. --opt perf_mode=THROUGHPUT (repeatable)")
    p.add_argument("--stats", action="store_true", help="Print speed stats")
    p.set_defaults(func=run_asr)
