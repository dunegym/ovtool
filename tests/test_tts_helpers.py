"""tts helpers: backend detection, per-backend property filtering, WAV
writing and speaker resolution."""
from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import openvino as ov
import pytest

from ovtool.tts import (_speech_props, detect_backend, resolve_speaker,
                        save_wav)


def test_detect_backend(tmp_path):
    (tmp_path / "kokoro" / "voices").mkdir(parents=True)
    assert detect_backend(str(tmp_path / "kokoro")) == "kokoro"
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "config.json").write_text('{"model_type": "Kokoro"}',
                                                  encoding="utf-8")
    assert detect_backend(str(tmp_path / "cfg")) == "kokoro"
    (tmp_path / "s5").mkdir()
    assert detect_backend(str(tmp_path / "s5")) == "speecht5"


def args(**kw):
    base = dict(language=None, speed=1.0, minlenratio=None,
                maxlenratio=None, threshold=None,
                speaker=None, speaker_embedding=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_speech_props_kokoro_keeps_its_own():
    props = _speech_props(args(language="en-us", speed=1.5), "kokoro")
    assert props == {"language": "en-us", "speed": 1.5}


def test_speech_props_default_speed_dropped():
    assert _speech_props(args(language=None, speed=1.0), "kokoro") == {}


def test_speech_props_speecht5_keeps_its_own():
    props = _speech_props(args(maxlenratio=0.9, minlenratio=0.5), "speecht5")
    assert props == {"maxlenratio": 0.9, "minlenratio": 0.5}


def test_speech_props_mismatched_params_warn(capsys):
    _speech_props(args(minlenratio=0.5), "kokoro")
    assert "applies to SpeechT5" in capsys.readouterr().out
    _speech_props(args(speed=1.5), "speecht5")
    assert "applies to Kokoro" in capsys.readouterr().out


def test_speech_props_unsupported_kokoro_language_warns(capsys):
    _speech_props(args(language="zh"), "kokoro")
    assert "not in the" in capsys.readouterr().out


class _Result:
    speeches = [ov.Tensor(np.array([[0.1, -0.5, 1.5, -1.5]], dtype=np.float32))]
    output_sample_rate = 16000


def test_save_wav_writes_int16_pcm(tmp_path):
    path, duration = save_wav(_Result(), tmp_path / "out.wav")
    assert duration == pytest.approx(4 / 16000)
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == 16000 and w.getnframes() == 4
        pcm = np.frombuffer(w.readframes(4), dtype=np.int16)
    assert pcm[0] == pytest.approx(3276, abs=2)
    assert pcm[1] == pytest.approx(-16383, abs=2)
    assert pcm[2] == 32767 and pcm[3] == -32767   # clipped to int16 range


class _Pipe:
    def __init__(self, shape=(512,)):
        self.shape = shape

    def get_speaker_embedding_shape(self):
        return self.shape


@pytest.fixture
def kokoro_voices(tmp_path):
    model = tmp_path / "kokoro"
    (model / "voices").mkdir(parents=True)
    for name in ("af_heart", "af_bella"):
        np.arange(512, dtype=np.float32).tofile(model / "voices" / f"{name}.bin")
    return model


def test_resolve_speaker_defaults_to_first_voice(kokoro_voices, capsys):
    tensor = resolve_speaker(_Pipe(), str(kokoro_voices), args())
    assert "af_bella" in capsys.readouterr().out       # sorted first
    assert list(tensor.shape) == [512]


def test_resolve_speaker_named_voice_case_insensitive(kokoro_voices):
    tensor = resolve_speaker(_Pipe(), str(kokoro_voices), args(speaker="AF_HEART"))
    assert np.asarray(tensor.data)[:3] == pytest.approx([0.0, 1.0, 2.0])


def test_resolve_speaker_unknown_voice_lists_options(kokoro_voices):
    with pytest.raises(SystemExit, match="unknown voice.*af_bella"):
        resolve_speaker(_Pipe(), str(kokoro_voices), args(speaker="nope"))


def test_resolve_speaker_embedding_file_overrides(kokoro_voices, tmp_path):
    emb = tmp_path / "custom.bin"
    np.full(512, 0.25, dtype=np.float32).tofile(emb)
    tensor = resolve_speaker(_Pipe(), str(kokoro_voices),
                             args(speaker_embedding=str(emb)))
    assert np.asarray(tensor.data)[0] == pytest.approx(0.25)


def test_resolve_speaker_shape_mismatch(kokoro_voices, tmp_path):
    emb = tmp_path / "custom.bin"
    np.full(512, 0.25, dtype=np.float32).tofile(emb)
    with pytest.raises(SystemExit, match="does not match"):
        resolve_speaker(_Pipe(shape=(256,)), str(kokoro_voices),
                        args(speaker_embedding=str(emb)))


def test_resolve_speaker_no_voices_returns_none(tmp_path):
    (tmp_path / "s5").mkdir()
    assert resolve_speaker(_Pipe(), str(tmp_path / "s5"), args()) is None
