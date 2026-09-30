"""KBStore / KnowledgeBase: creation validation, document replacement,
index building and deletion — pure filesystem, no pipelines."""
from __future__ import annotations

import numpy as np
import pytest

from ovtool.rag import KBError, KBStore
from tests.conftest import make_model


@pytest.fixture
def store_with_embedder(tmp_path):
    embedder = make_model(tmp_path / "embedders" / "FakeEmb" / "int8", "embed")
    store = KBStore(tmp_path / "kb")

    def new_kb(name="kb", **kw):
        # store.create() returns the summary dict; tests want the KB object
        return store.get(store.create(name, str(embedder), **kw)["id"])
    return store, embedder, new_kb


def vecs(rows, dim=3):
    return np.ones((rows, dim), dtype=np.float32)


def test_create_validates_name_and_embedder(store_with_embedder):
    store, embedder, new_kb = store_with_embedder
    with pytest.raises(KBError):
        new_kb("")
    with pytest.raises(KBError):
        new_kb("x" * 81)
    with pytest.raises(KBError, match="not an embedding model"):
        store.create("ok", str(embedder.parent))  # not a model dir


def test_create_validates_chunking(store_with_embedder):
    store, embedder, new_kb = store_with_embedder
    with pytest.raises(KBError, match="chunk size"):
        new_kb("kb", chunk_size=50)
    with pytest.raises(KBError, match="overlap"):
        new_kb("kb", chunk_size=400, chunk_overlap=101)
    assert new_kb("kb", chunk_size=400, chunk_overlap=100)


def test_create_rejects_duplicate_names_case_insensitively(store_with_embedder):
    store, embedder, new_kb = store_with_embedder
    new_kb("Docs")
    with pytest.raises(KBError, match="already exists"):
        new_kb("docs")


def test_get_unknown_kb_is_404(store_with_embedder):
    store, _, _ = store_with_embedder
    with pytest.raises(KBError) as ei:
        store.get("nosuchkb")
    assert ei.value.status == 404


def test_add_document_and_index_roundtrip(store_with_embedder):
    store, _, new_kb = store_with_embedder
    kb = new_kb()
    store.add_document(kb.id, name="a.txt", chunks=["alpha one", "alpha two"],
                       vectors=vecs(2), source="paste", sha1="x", chars=20)
    store.add_document(kb.id, name="b.txt", chunks=["beta one"],
                       vectors=vecs(1), source="paste", sha1="y", chars=9)
    summary = next(s for s in store.list() if s["name"] == "kb")
    assert summary["chunks"] == 3 and summary["chars"] == 29
    matrix, rows, texts, names = kb.index()
    assert matrix.shape == (3, 3)
    assert len(rows) == 3
    doc_a = kb.meta["docs"][0]["id"]
    assert texts[doc_a] == ["alpha one", "alpha two"]
    assert set(names.values()) == {"a.txt", "b.txt"}


def test_add_document_replaces_same_name(store_with_embedder):
    store, _, new_kb = store_with_embedder
    kb = new_kb()
    first = store.add_document(kb.id, name="a.txt", chunks=["old"],
                               vectors=vecs(1), source="paste", sha1="x", chars=3)
    store.add_document(kb.id, name="a.txt", chunks=["new1", "new2"],
                       vectors=vecs(2), source="paste", sha1="x2", chars=8)
    docs = kb.meta["docs"]
    assert len(docs) == 1 and docs[0]["chunks"] == 2
    assert not (kb.root / "docs" / f"{first['id']}.npy").exists()


def test_add_document_rejects_dim_change(store_with_embedder):
    store, _, new_kb = store_with_embedder
    kb = new_kb()
    store.add_document(kb.id, name="a.txt", chunks=["x"], vectors=vecs(1, 3),
                       source="paste", sha1="x", chars=1)
    with pytest.raises(KBError, match="dim"):
        store.add_document(kb.id, name="b.txt", chunks=["y"], vectors=vecs(1, 4),
                           source="paste", sha1="y", chars=1)


def test_index_skips_corrupt_doc_files(store_with_embedder):
    store, _, new_kb = store_with_embedder
    kb = new_kb()
    store.add_document(kb.id, name="a.txt", chunks=["alpha"], vectors=vecs(1),
                       source="paste", sha1="x", chars=5)
    bad = store.add_document(kb.id, name="b.txt", chunks=["beta"], vectors=vecs(1),
                             source="paste", sha1="y", chars=4)
    (kb.root / "docs" / f"{bad['id']}.npy").write_bytes(b"not an npy file")
    kb.invalidate()
    matrix, rows, texts, names = kb.index()
    assert matrix.shape[0] == 1 and len(rows) == 1
    assert set(texts) == {kb.meta["docs"][0]["id"]}   # only the intact doc loads


def test_delete_document(store_with_embedder):
    store, _, new_kb = store_with_embedder
    kb = new_kb()
    doc = store.add_document(kb.id, name="a.txt", chunks=["x"], vectors=vecs(1),
                             source="paste", sha1="x", chars=1)
    with pytest.raises(KBError):
        store.delete_document(kb.id, "nosuchdoc")
    store.delete_document(kb.id, doc["id"])
    assert kb.meta["docs"] == []
    assert not (kb.root / "docs" / f"{doc['id']}.json").exists()


def test_delete_kb_removes_directory(store_with_embedder):
    store, _, new_kb = store_with_embedder
    kb = new_kb()
    assert kb.root.is_dir()
    store.delete(kb.id)
    assert not kb.root.exists()
    assert store.list() == []


def test_kb_error_carries_http_status():
    e = KBError("nope", 409)
    assert e.status == 409
    assert KBError("bad").status == 400
