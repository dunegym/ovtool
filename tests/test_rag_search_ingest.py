"""rag.search() and the background Ingestor, driven by the stub retriever —
no real embedding pipeline anywhere."""
from __future__ import annotations

import shutil
import threading
import time

import pytest

from ovtool import rag
from tests.conftest import FakeRetriever

DOCS = [("alpha.txt", "alpha material"),
        ("beta.txt", "beta material"),
        ("gamma.txt", "gamma material")]


def wait_for(cond, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_search_ranks_by_cosine(kb_factory):
    store, kb = kb_factory(DOCS)
    res = rag.search(store, FakeRetriever(), kb.id, "alpha", top_k=2, top_n=2)
    assert res["candidates"][0]["cosine"] == pytest.approx(1.0)
    assert "alpha" in res["candidates"][0]["text"]
    assert len(res["candidates"]) == 2
    assert res["best"] == pytest.approx(1.0)
    assert "embed_ms" in res["timings"] and "search_ms" in res["timings"]


def test_search_min_score_filters_without_reranker(kb_factory):
    store, kb = kb_factory(DOCS)
    res = rag.search(store, FakeRetriever(), kb.id, "alpha",
                     top_k=3, top_n=3, min_score=0.5)
    hits = res["hits"]
    assert len(hits) == 1 and "alpha" in hits[0]["text"]
    assert all(h["kept"] for h in hits)
    assert any(not c["kept"] for c in res["candidates"])


def test_search_reranker_reorders_and_filters(kb_factory):
    store, kb = kb_factory(DOCS)
    stages = []
    res = rag.search(store, FakeRetriever(), kb.id, "alpha", top_k=3, top_n=3,
                     reranker="models/rerank/fake", min_score=0.5,
                     on_stage=lambda s, d=None: stages.append(s))
    assert "rerank" in stages
    # the stub reranker scores alpha-matching texts 0.9, others 0.2
    assert all(c["rerank"] == pytest.approx(0.9) for c in res["candidates"]
               if "alpha" in c["text"])
    assert len(res["hits"]) == 1
    assert res["hits"][0]["score"] == pytest.approx(0.9)


def test_search_empty_kb_is_rejected(kb_factory):
    store, kb = kb_factory([])
    with pytest.raises(rag.KBError, match="no documents"):
        rag.search(store, FakeRetriever(), kb.id, "alpha")


def test_search_missing_embedder_dir_is_rejected(kb_factory, tmp_path):
    store, kb = kb_factory(DOCS)
    shutil.rmtree(tmp_path / "embedders0", ignore_errors=True)
    with pytest.raises(rag.KBError, match="not found"):
        rag.search(store, FakeRetriever(), kb.id, "alpha")


def test_search_dim_mismatch_is_rejected(kb_factory, monkeypatch):
    store, kb = kb_factory(DOCS)
    retr = FakeRetriever()

    def short_embed(profile, device, texts, *, progress=None, cancelled=None):
        import numpy as np
        return np.zeros((len(texts), 2), np.float32)

    monkeypatch.setattr(retr, "embed", short_embed)
    with pytest.raises(rag.KBError, match="dim"):
        rag.search(store, retr, kb.id, "alpha")


# ---------------- Ingestor ---------------- #

def test_ingestor_processes_pastes_end_to_end(kb_factory):
    store, kb = kb_factory([])
    embedder = kb.meta["embedder"]["path"]
    ing = rag.Ingestor(store, FakeRetriever())

    assert ing.submit(kb.id, [{"name": "doc.txt", "text": "alpha notes",
                               "source": "paste"}], "CPU") == 1
    assert wait_for(lambda: ing.status()["recent"]
                    and "error" not in ing.status()["recent"][0])
    recent = ing.status()["recent"][0]
    assert recent["chunks"] == 1 and recent.get("skipped") is None

    docs = store.get(kb.id).meta["docs"]
    assert len(docs) == 1 and docs[0]["name"] == "doc.txt"
    assert kb.meta["embedder"]["path"] == embedder  # KB stays bound to it


def test_ingestor_skips_unchanged_and_replaces_changed(kb_factory):
    store, kb = kb_factory([])
    ing = rag.Ingestor(store, FakeRetriever())
    item = {"name": "doc.txt", "text": "alpha one", "source": "paste"}

    ing.submit(kb.id, [item], "CPU")
    assert wait_for(lambda: ing.status()["recent"])
    ing.submit(kb.id, [item], "CPU")  # identical content
    assert wait_for(lambda: len(ing.status()["recent"]) >= 2)
    assert ing.status()["recent"][0].get("skipped") is True

    changed = {**item, "text": "alpha one and two"}
    ing.submit(kb.id, [changed], "CPU")
    assert wait_for(lambda: len(ing.status()["recent"]) >= 3)
    recent = ing.status()["recent"][0]
    assert recent.get("replaced") is True and recent["chunks"] == 1
    assert len(store.get(kb.id).meta["docs"]) == 1


def test_ingestor_cancel_drops_queue_and_stops_current(kb_factory):
    store, kb = kb_factory([])
    gate = threading.Event()
    ing = rag.Ingestor(store, FakeRetriever(block_event=gate))
    items = [{"name": f"d{i}.txt", "text": f"alpha {i}", "source": "paste"}
             for i in range(3)]

    ing.submit(kb.id, items, "CPU")
    assert wait_for(lambda: ing.status()["current"] is not None)
    dropped = ing.cancel()
    assert dropped == 2
    assert wait_for(lambda: ing.status()["current"] is None
                    and ing.status()["recent"])
    recent = ing.status()["recent"][0]
    assert recent.get("error") == "cancelled"
    assert ing.status()["queued"] == 0
    assert store.get(kb.id).meta["docs"] == []
    gate.set()  # release the worker should it re-enter embed


def test_collect_files_walks_folders_and_skips_noise(tmp_path):
    (tmp_path / "proj" / "sub").mkdir(parents=True)
    (tmp_path / "proj" / "node_modules").mkdir()
    (tmp_path / "proj" / ".git").mkdir()
    (tmp_path / "proj" / "root.md").write_text("root", encoding="utf-8")
    (tmp_path / "proj" / "sub" / "note.txt").write_text("note", encoding="utf-8")
    (tmp_path / "proj" / "binary.bin").write_bytes(b"\0\1")
    (tmp_path / "proj" / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
    (tmp_path / "proj" / ".git" / "cfg").write_text("x", encoding="utf-8")

    items = rag.collect_files(str(tmp_path / "proj"))
    names = sorted(i["name"] for i in items)
    assert names == ["proj/root.md", "proj/sub/note.txt"]
    assert all(i["source"] == "path" for i in items)


def test_collect_files_single_file_and_errors(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi", encoding="utf-8")
    assert rag.collect_files(str(f)) == [
        {"name": "a.txt", "path": str(f.resolve()), "source": "path"}]
    with pytest.raises(rag.KBError, match="not a file or folder"):
        rag.collect_files(str(tmp_path / "missing"))
