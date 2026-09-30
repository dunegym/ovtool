"""webui server logic without sockets: the model slot, load-state gating,
generation-lock timeout, settings persistence, root merging and the catalog
scan. Locks the fixes for the unreachable 500 branch and the un-timed
generation lock (regression tests for commit d7f2969)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import openvino as ov
import pytest

import ovtool.webui as W
from tests.conftest import make_model


class RecordingHandler(W.Handler):
    """Handler with the HTTP plumbing stubbed out; records SSE/JSON output."""

    def __init__(self):
        self.events = []
        self.json = []
        self.started = self.ended = False

    def log_message(self, fmt, *args):
        pass

    def log_error(self, fmt, *args):
        pass

    def _json(self, status, obj):
        self.json.append((status, obj))

    def _sse_start(self):
        self.started = True

    def _sse(self, obj):
        self.events.append(obj)
        return True

    def _sse_end(self):
        self.ended = True


@pytest.fixture(autouse=True)
def clean_slot():
    W.SLOT.reset()
    yield
    W.SLOT.reset()


# ---------------- ModelSlot ---------------- #

def test_model_slot_reset_and_describe():
    W.SLOT.pipe = object()
    W.SLOT.kind = "llm"
    W.SLOT.path = "some/model/dir"
    W.SLOT.devices = ["NPU", "NPU", "GPU"]
    d = W.SLOT.describe()
    assert d["loaded"] and d["name"] == "dir" and d["devices"] == ["NPU", "NPU", "GPU"]
    W.SLOT.reset()
    d = W.SLOT.describe()
    assert not d["loaded"] and d["path"] is None and d["devices"] is None


# ---------------- _require_chat ordering (fixed: was unreachable 500) ----- #

def test_require_chat_loading_is_409():
    W.SLOT.loading = True
    with pytest.raises(W._ApiError) as ei:
        RecordingHandler()._require_chat()
    assert ei.value.status == 409


def test_require_chat_failed_load_is_500_with_reason():
    W.SLOT.error = "boom: compile failed"
    with pytest.raises(W._ApiError) as ei:
        RecordingHandler()._require_chat()
    assert ei.value.status == 500 and "boom" in ei.value.message


def test_require_chat_empty_slot_is_400():
    with pytest.raises(W._ApiError) as ei:
        RecordingHandler()._require_chat()
    assert ei.value.status == 400


def test_require_chat_loaded_chat_kind_passes():
    W.SLOT.pipe = object()
    W.SLOT.kind = "vlm"
    RecordingHandler()._require_chat()


def test_require_chat_loaded_image_kind_is_400():
    W.SLOT.pipe = object()
    W.SLOT.kind = "image"
    with pytest.raises(W._ApiError) as ei:
        RecordingHandler()._require_chat()
    assert ei.value.status == 400


# ---------------- generation-lock timeout (fixed: constant was unused) ---- #

class BoomPipe:
    def generate(self, *a, **k):
        raise RuntimeError("budget exceeded")

    def get_tokenizer(self):
        raise RuntimeError("no tokenizer")


@pytest.fixture
def loaded_llm(monkeypatch):
    monkeypatch.setattr(W, "GENERATION_LOCK_TIMEOUT", 0.2)
    W.SLOT.pipe = BoomPipe()
    W.SLOT.kind = "llm"
    W.SLOT.tokenizer = None


def test_chat_nonstream_lock_timeout_is_409(loaded_llm):
    h = RecordingHandler()
    W.SLOT.gen_lock.acquire()  # same-thread Lock: acquire() times out, no deadlock
    try:
        with pytest.raises(W._ApiError) as ei:
            h._chat({"messages": [{"role": "user", "content": "hi"}],
                     "stream": False, "params": {"max_tokens": 4}})
        assert ei.value.status == 409
        assert "still running" in ei.value.message
    finally:
        W.SLOT.gen_lock.release()


def test_chat_nonstream_failure_releases_lock(loaded_llm):
    h = RecordingHandler()
    with pytest.raises(W._ApiError) as ei:
        h._chat({"messages": [{"role": "user", "content": "hi"}],
                 "stream": False, "params": {"max_tokens": 4}})
    assert ei.value.status == 500 and "generation failed" in ei.value.message
    assert W.SLOT.gen_lock.acquire(blocking=False)   # released by finally
    W.SLOT.gen_lock.release()


def test_chat_stream_lock_timeout_reports_in_band(loaded_llm):
    h = RecordingHandler()
    W.SLOT.gen_lock.acquire()
    try:
        h._chat({"messages": [{"role": "user", "content": "hi"}],
                 "stream": True, "params": {"max_tokens": 4}})
    finally:
        W.SLOT.gen_lock.release()
    assert h.started and h.ended
    assert any(e and "still running" in e.get("error", "") for e in h.events)
    assert h.events[-1] is None          # terminal [DONE]


# ---------------- settings persistence ---------------- #

@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "webui_settings.json"
    monkeypatch.setattr(W, "SETTINGS_PATH", path)
    monkeypatch.setattr(W, "SESSION_ROOTS", [])
    return path


def test_save_and_load_settings_roundtrip(settings_file):
    W.save_settings({"theme": "light", "lang": "zh"})
    assert W.load_settings() == {"theme": "light", "lang": "zh"}


def test_load_settings_corrupt_file_falls_back(settings_file):
    settings_file.write_text("{broken", encoding="utf-8")
    assert W.load_settings() == {}


def test_persist_now_writes_or_drops_file(settings_file):
    h = RecordingHandler()
    h.settings = {"theme": "dark", "persist": True}
    h._persist_now()
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert data["persist"] is True and data["roots"] == []
    h.settings["persist"] = False
    h._persist_now()
    assert not settings_file.exists()


def test_update_settings_validates(settings_file):
    h = RecordingHandler()
    h.settings = dict(W.DEFAULT_SETTINGS)
    with pytest.raises(W._ApiError):
        h._update_settings({"theme": "blue"})
    with pytest.raises(W._ApiError):
        h._update_settings({"lang": "fr"})
    with pytest.raises(W._ApiError):
        h._update_settings({"ui": "not an object"})
    h._update_settings({"theme": "light", "ui": {"p-max": "64"}})
    assert h.settings["theme"] == "light"
    assert h.settings["ui"]["p-max"] == "64"


# ---------------- model roots & catalog ---------------- #

def test_all_roots_dedups_and_keeps_default_first(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(W, "SESSION_ROOTS", [str(models), str(other)])
    monkeypatch.setenv("OVTOOL_MODELS_PATH", str(other))  # dup of a session root
    roots = W.all_roots(str(models))
    assert roots[0] == str(models)
    resolved = [str(Path(r).resolve()) for r in roots]
    assert resolved.count(str(other.resolve())) == 1
    assert resolved.count(str(models.resolve())) == 1


def test_scan_models_groups_kinds_and_disambiguates_names(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    make_model(a / "llm" / "M" / "int4", "llm")
    make_model(a / "image" / "SD" / "int8", "image")
    make_model(a / "tts" / "S" / "fp16", "tts")        # not loadable: excluded
    make_model(b / "llm" / "M" / "int8", "llm")        # same name, other root
    monkeypatch.setattr(W, "SESSION_ROOTS", [str(b)])
    groups = W.scan_models(str(a))
    assert {g["kind"] for g in groups} == {"llm", "image"}
    llm_groups = [g for g in groups if g["kind"] == "llm"]
    assert sum(len(g["variants"]) for g in llm_groups) == 2
    assert any("M (" in g["name"] for g in llm_groups)  # cross-root disambiguation


def test_resolve_eq(tmp_path):
    f = tmp_path / "x"
    f.mkdir()
    assert W._resolve_eq(str(f), str(tmp_path / "x"))
    assert not W._resolve_eq(str(f), str(tmp_path / "y"))


# ---------------- misc helpers ---------------- #

def test_b64_png_encodes_real_png():
    arr = np.zeros((1, 8, 8, 3), np.uint8)
    urls = W._b64_png(ov.Tensor(arr))
    assert len(urls) == 1 and urls[0].startswith("data:image/png;base64,")
    assert base64.b64decode(urls[0].split(",", 1)[1])[:4] == b"\x89PNG"


def test_rag_options_validation(tmp_path):
    from ovtool import rag
    store = rag.KBStore(tmp_path / "kb")
    embedder = make_model(tmp_path / "e" / "FakeEmb" / "int8", "embed")
    kb_id = store.create("kb", str(embedder))["id"]
    h = RecordingHandler()
    h.kb = store
    opts = h._rag_options({"kb": kb_id, "top_k": 10, "top_n": 3})
    assert opts["top_k"] == 10 and opts["top_n"] == 3
    for bad in ({"kb": kb_id, "top_n": 0}, {"kb": kb_id, "top_n": 30},
                {"kb": kb_id, "top_k": 2, "top_n": 4}):
        with pytest.raises(W._ApiError):
            h._rag_options(bad)
    with pytest.raises(rag.KBError):
        h._rag_options({"kb": "nosuch"})
    assert h._rag_options({"kb": None}) is None       # RAG off
