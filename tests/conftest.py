"""Shared fixtures: fabricated OpenVINO model directories and stub retrievers.

Nothing here needs a real model or an OpenVINO device — the suite exercises
ovtool's own logic (classification, registry gating, chunking, KB storage,
retrieval, HTTP handlers) against the smallest directory layouts that
registry.detect_kind() recognizes.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

# the stub embedder's vocabulary: a text's vector is the normalized sum of
# the basis vectors of the keywords it contains
KEYWORDS = ("alpha", "beta", "gamma")


def make_model(dir_, kind, config=None):
    """Fabricate the smallest directory that detect_kind() classifies as kind."""
    dir_.mkdir(parents=True, exist_ok=True)

    def w(name, text):
        (dir_ / name).write_text(text, encoding="utf-8")

    if kind == "image":
        w("model_index.json", "{}")
    elif kind == "vlm":
        w("openvino_text_embeddings_model.xml", "<net/>")
    elif kind == "tts":  # SpeechT5-style component IR
        w("openvino_postnet.xml", "<net/>")
    elif kind == "kokoro":
        w("openvino_model.xml", "<net/>")
        w("config.json", json.dumps({"model_type": "kokoro"}))
    elif kind == "embed":
        w("openvino_model.xml", "<net/>")
        w("config.json", json.dumps({"architectures": ["BertModel"], **(config or {})}))
    elif kind == "rerank":
        w("openvino_model.xml", "<net/>")
        w("config.json", json.dumps(
            {"architectures": ["XLMRobertaForSequenceClassification"], **(config or {})}))
    elif kind == "llm":
        w("openvino_language_model.xml", "<net/>")
        w("config.json", json.dumps(
            {"architectures": ["Qwen3ForCausalLM"], **(config or {})}))
    else:
        raise ValueError(kind)
    return dir_


@pytest.fixture
def make_model_dir(tmp_path):
    def _make(kind, name="model", config=None):
        return make_model(tmp_path / name, kind, config)
    return _make


def fake_vector(text, dim=len(KEYWORDS)):
    v = np.zeros(dim, np.float32)
    low = text.lower()
    for i, kw in enumerate(KEYWORDS):
        if kw in low:
            v[i] = 1.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


class FakeRetriever:
    """Stands in for rag.Retriever: keyword-basis embeddings and rule-based
    rerank scores, so rag.search() and rag.Ingestor run with no pipelines.

    With block_event set, embed() parks until the event is set or the caller
    reports cancellation (the Ingestor cancel test)."""

    def __init__(self, block_event=None):
        self.block_event = block_event
        self.embed_calls = 0

    def is_loaded(self, kind, path, device):
        return True

    def ensure(self, kind, path, device, pooling=None):
        raise AssertionError("search() must not load models when is_loaded() is True")

    def embed(self, profile, device, texts, *, progress=None, cancelled=None):
        out = []
        for i, t in enumerate(texts):
            if self.block_event is not None:
                while not self.block_event.is_set():
                    if cancelled is not None and cancelled():
                        from ovtool.rag import KBError
                        raise KBError("cancelled", 409)
                    time.sleep(0.005)
            out.append(fake_vector((profile.get("query_instruction") or "") + t))
            self.embed_calls += 1
            if progress is not None:
                progress(i + 1, len(texts))
        return np.asarray(out, dtype=np.float32)

    def rerank(self, path, device, query, texts):
        kws = [kw for kw in KEYWORDS if kw in query.lower()]
        return [0.9 if any(kw in t.lower() for kw in kws) else 0.2 for t in texts]


@pytest.fixture
def kb_factory(tmp_path):
    """Build a KBStore with one knowledge base whose documents are embedded
    with the stub embedder; docs is [(name, text)]."""
    from ovtool import rag

    counter = iter(range(1000))

    def _build(docs, name="test"):
        n = next(counter)
        embedder = make_model(
            tmp_path / f"embedders{n}" / "FakeEmb" / "int8", "embed")
        store = rag.KBStore(tmp_path / f"kb{n}")
        kb = store.get(store.create(name, str(embedder))["id"])
        for doc_name, text in docs:
            chunks = rag.chunk_text(text)
            vecs = np.stack([fake_vector(c) for c in chunks])
            store.add_document(kb.id, name=doc_name, chunks=chunks, vectors=vecs,
                               source="paste", sha1=doc_name, chars=len(text))
        return store, kb
    return _build
