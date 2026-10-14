"""asr helpers: WAV loading (downmix + resample to the 16 kHz mono float32
the pipelines expect)."""
from __future__ import annotations

import wave

import numpy as np
import pytest

from ovtool.asr import TARGET_RATE, read_wav_16k_mono


def _write_wav(path, samples, rate, nch=1, sampwidth=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        if sampwidth == 2:
            data = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
            if nch > 1:
                data = np.stack([data] * nch, axis=1)
            w.writeframes(data.tobytes())
        else:  # 32-bit float
            raw = samples.astype(np.float32)
            if nch > 1:
                raw = np.stack([raw] * nch, axis=1)
            w.writeframes(raw.tobytes())


def test_passthrough_16k_mono(tmp_path):
    f = tmp_path / "a.wav"
    _write_wav(f, np.linspace(-0.5, 0.5, 1600, dtype=np.float32), 16000)
    out, rate = read_wav_16k_mono(str(f))
    assert rate == 16000 and out.shape == (1600,) and out.dtype == np.float32
    assert out.max() <= 1.0 and abs(float(np.abs(out).max()) - 0.5) < 1e-3


def test_stereo_downmix_and_resample(tmp_path):
    f = tmp_path / "b.wav"
    _write_wav(f, np.full(9600, 0.25, dtype=np.float32), 48000, nch=2)
    out, rate = read_wav_16k_mono(str(f))
    assert rate == 48000                      # original rate reported
    assert out.shape == (3200,)               # 0.2 s at 48k -> 16k
    assert np.allclose(out, 0.25, atol=0.01)  # downmix averaged equal channels


def test_float32_wav(tmp_path):
    f = tmp_path / "c.wav"
    _write_wav(f, np.full(8000, -0.75, dtype=np.float32), 16000, sampwidth=4)
    out, _ = read_wav_16k_mono(str(f))
    assert out.shape == (8000,) and np.allclose(out, -0.75, atol=1e-6)


def test_unsupported_width(tmp_path):
    f = tmp_path / "d.wav"
    _write_wav(f, np.zeros(100, dtype=np.float32), 16000)
    data = bytearray(f.read_bytes())
    data[34] = 1                               # sampwidth byte in the fmt chunk
    g = tmp_path / "e.wav"
    g.write_bytes(bytes(data))
    with pytest.raises(SystemExit, match="sample width"):
        read_wav_16k_mono(str(g))
