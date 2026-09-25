"""Knowledge bases for retrieval-augmented chat (the web UI's RAG).

A knowledge base (KB) is a directory under the KB root (default
~/.ovtool/kb, `ovtool webui --kb-dir`):

  <kb>/kb.json          name, embedding-model profile (path, pooling, query
                        instruction, dim), chunking parameters, documents
  <kb>/docs/<doc>.json  the document's chunk texts
  <kb>/docs/<doc>.npy   float32 [chunks, dim] unit-length embeddings

Vectors of different embedding models (even the fp16 and int8 exports of
one model) are not comparable, so a KB stays bound to the embedder it was
created with. Retrieval is exact: cosine similarity (a dot product of unit
vectors) against every chunk; the best candidates are re-scored by an
optional cross-encoder reranker and the top passages are spliced into the
user's turn as numbered, citable references.

Pipelines follow the `ovtool embed` / `ovtool rerank` conventions: pooling
from the sentence-transformers 1_Pooling config (last-token for Qwen3), the
model family's query instruction, and the official Qwen3-Reranker template.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from collections import OrderedDict, deque
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .embed import (QWEN3_QUERY_INSTRUCT, _pooling_enum, is_qwen3,
                    qwen3_rerank_wrap, read_pooling)

DEFAULT_KB_DIR = Path.home() / ".ovtool" / "kb"
DEFAULT_CHUNK_SIZE = 800      # characters
DEFAULT_CHUNK_OVERLAP = 120
EMBED_BATCH = 16              # chunks per embed call; ingestion yields between batches
MAX_DOC_CHARS = 5_000_000
MAX_FILE_BYTES = 100 << 20
MAX_FOLDER_FILES = 5000


class KBError(Exception):
    """A knowledge-base request that cannot be served (carries an HTTP status)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _model_name(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


# --------------------------------------------------------------------------- #
# text extraction
# --------------------------------------------------------------------------- #

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".adoc", ".tex", ".log", ".srt",
    ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".vue", ".java", ".kt", ".scala",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php",
    ".swift", ".lua", ".r", ".sql", ".sh", ".bat", ".ps1", ".css", ".scss",
}
MARKUP_EXTS = {".html", ".htm", ".xhtml"}
SUPPORTED_EXTS = TEXT_EXTS | MARKUP_EXTS | {".pdf", ".docx"}


def _decode(data: bytes) -> str:
    """Text-file bytes -> str: UTF-16 by BOM, UTF-8 (BOM optional), then
    GB18030 (GBK-encoded Chinese files are common on Windows)."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")


class _HTMLText(HTMLParser):
    """Visible text of an HTML page, with line breaks at block elements."""

    SKIP = {"script", "style", "noscript", "template", "svg"}
    BLOCK = {"p", "div", "br", "hr", "li", "ul", "ol", "dl", "dt", "dd", "tr",
             "table", "section", "article", "header", "footer", "nav", "aside",
             "main", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote",
             "figcaption", "title"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(re.sub(r"\s+", " ", data))  # HTML collapses whitespace


def _html_text(markup: str) -> str:
    parser = _HTMLText()
    parser.feed(markup)
    parser.close()
    return "\n".join(line.strip() for line in "".join(parser.parts).split("\n"))


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(data: bytes) -> str:
    """Paragraph text of a .docx (body, text boxes; table rows as
    "cell | cell" lines) with the stdlib."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if z.getinfo("word/document.xml").file_size > 200 << 20:
                raise KBError("document.xml larger than 200 MB")
            root = ElementTree.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
        raise KBError(f"not a readable .docx file: {e}") from None
    out: list[str] = []

    def walk(el) -> None:  # recursive, so nested paragraphs are not repeated
        if el.tag == _W + "t":
            out.append(el.text or "")
        elif el.tag == _W + "tab":
            out.append("\t")
        elif el.tag in (_W + "br", _W + "cr"):
            out.append("\n")
        for child in el:
            walk(child)
        if el.tag == _W + "p":
            out.append("\n")
        elif el.tag == _W + "tc":  # a cell ends in " | " instead of a newline
            while out and out[-1] == "\n":
                out.pop()
            out.append(" | ")
        elif el.tag == _W + "tr":
            if out and out[-1] == " | ":
                out.pop()
            out.append("\n")

    walk(root)
    return "".join(out)


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise KBError("PDF files need pypdf: pip install pypdf") from None
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:  # noqa: BLE001 - pypdf raises many types
        raise KBError(f"could not read PDF: {e}") from None


def normalize_text(text: str) -> str:
    """Unify newlines; drop NULs, BOMs, trailing spaces and runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    text = text.replace("\x00", "").replace("﻿", "")
    text = re.sub(r"[ \t\v 　]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_text(name: str, data: bytes) -> str:
    """Plain text of a document, dispatched on its file extension; unknown
    extensions are accepted when the content does not look binary."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        text = _pdf_text(data)
    elif ext == ".docx":
        text = _docx_text(data)
    elif ext in MARKUP_EXTS:
        text = _html_text(_decode(data))
    elif ext in TEXT_EXTS or b"\0" not in data[:8192]:
        text = _decode(data)
    else:
        raise KBError(f"unsupported file type: {ext or name}")
    return normalize_text(text)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #

# boundary levels, strongest first, as (separator, how many of its chars
# stay with the chunk on the left): before a markdown heading, then after
# a paragraph, line, sentence (CJK and Latin), clause, word
_LEVELS = (
    (("\n#", 1),),
    (("\n\n", 2),),
    (("\n", 1),),
    (("。", 1), ("！", 1), ("？", 1), ("…", 1), (". ", 2), ("! ", 2), ("? ", 2)),
    (("；", 1), ("; ", 2), ("：", 1), (": ", 2), ("，", 1), ("、", 1), (", ", 2)),
    ((" ", 1), ("\t", 1)),
)
# bump when chunking changes, so re-adding an unchanged file re-chunks it
CHUNKER_VERSION = 2


def _cut(text: str, lo: int, hi: int) -> int:
    """End of a chunk: at the last boundary of the strongest level present
    in text[lo:hi]; hi (a hard cut) when there is none."""
    for level in _LEVELS:
        best = -1
        for sep, keep in level:
            i = text.rfind(sep, lo, hi)
            if i >= 0:
                best = max(best, i + keep)
        if best > lo:
            return best
    return hi


def _resume(text: str, lo: int, hi: int) -> int:
    """Start of the next chunk inside the overlap text[lo:hi]: at the first
    boundary of the strongest level present, so the carried-over context
    opens on a heading, paragraph, sentence or word; lo when there is none."""
    for level in _LEVELS:
        best = hi
        for sep, keep in level:
            i = text.find(sep, lo, hi)
            if i >= 0:
                best = min(best, i + keep)
        if best < hi:
            return best
    return lo


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of at most `size` characters, each cut at the
    strongest boundary in the back half of its window (markdown sections
    stay together where they fit); consecutive chunks share up to `overlap`
    characters, except that a new section starts clean. overlap <= size / 4
    keeps every step at least a quarter chunk forward."""
    text = normalize_text(text)
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = n if n - start <= size else _cut(text, start + size // 2, start + size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        if not overlap or text.startswith("#", end):
            start = end
        else:
            start = _resume(text, end - overlap, end)
    return chunks


# --------------------------------------------------------------------------- #
# embedding-model profile
# --------------------------------------------------------------------------- #

def default_instructions(model_dir: str) -> tuple[str | None, str | None]:
    """(query prefix, document prefix) recommended by the embedder's family.
    GenAI's query_instruction is a plain prefix, so prefixes are applied
    here and one resident pipeline can serve every knowledge base."""
    if is_qwen3(model_dir):  # model card format: Instruct: <task>\nQuery:<query>
        return f"Instruct: {QWEN3_QUERY_INSTRUCT}\nQuery:", None
    names = " ".join(Path(model_dir).parts[-3:]).lower()
    if "bge" in names and "m3" not in names:  # bge-m3 needs no instruction
        if re.search(r"bge[\w.-]*-zh", names):
            return "为这个句子生成表示以用于检索相关文章：", None
        return "Represent this sentence for searching relevant passages: ", None
    if re.search(r"(^|[\s_-])(multilingual-)?e5([\s_-]|$)", names):
        return "query: ", "passage: "
    return None, None


def embedder_profile(model_dir: str) -> dict:
    path = str(Path(model_dir).resolve())
    query_instr, doc_instr = default_instructions(path)
    pooling = read_pooling(path) or ("last_token" if is_qwen3(path) else "mean")
    return {"path": path, "name": _model_name(path), "pooling": pooling,
            "query_instruction": query_instr, "doc_instruction": doc_instr,
            "dim": None}  # known after the first document is embedded


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #

_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _write_json(path: Path, obj) -> None:
    """Atomic write (temp + replace): a crash never leaves a torn file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _write_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


class KnowledgeBase:
    """One KB: its metadata plus a lazily built in-memory index (every chunk
    vector stacked into one matrix, with the (doc, chunk) of each row)."""

    def __init__(self, root: Path, meta: dict) -> None:
        self.root, self.meta = root, meta
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple[str, int]] = []
        self._texts: dict[str, list[str]] = {}

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def name(self) -> str:
        return self.meta["name"]

    def summary(self) -> dict:
        docs = self.meta["docs"]
        return {**{k: self.meta[k] for k in ("id", "name", "created", "embedder",
                                             "chunk_size", "chunk_overlap")},
                "docs": docs, "chunks": sum(d["chunks"] for d in docs),
                "chars": sum(d["chars"] for d in docs),
                "embedder_ok": Path(self.meta["embedder"]["path"]).is_dir()}

    def doc(self, doc_id: str) -> dict | None:
        return next((d for d in self.meta["docs"] if d["id"] == doc_id), None)

    def doc_by_name(self, name: str) -> dict | None:
        return next((d for d in self.meta["docs"] if d["name"] == name), None)

    def index(self) -> tuple[np.ndarray, list[tuple[str, int]],
                             dict[str, list[str]], dict[str, str]]:
        """Snapshot the vectors, row map, chunk texts and doc names together.

        Search releases the store lock while embedding the query; ingest may
        unlink `{id}.json` in that window. Returning the texts here means
        search never re-opens those files.
        """
        if self._matrix is None:
            mats, rows, texts = [], [], {}
            for d in self.meta["docs"]:
                doc_id = d["id"]
                npy = self.root / "docs" / f"{doc_id}.npy"
                js = self.root / "docs" / f"{doc_id}.json"
                try:
                    m = np.load(npy)
                    chunks = json.loads(js.read_text(encoding="utf-8"))["chunks"]
                except (OSError, ValueError):
                    continue  # half-replaced doc; skip until the next rebuild
                texts[doc_id] = chunks
                mats.append(m)
                rows.extend((doc_id, i) for i in range(len(m)))
            dim = self.meta["embedder"].get("dim") or 1
            self._matrix = np.concatenate(mats) if mats else np.zeros((0, dim), np.float32)
            self._rows, self._texts = rows, texts
        names = {d["id"]: d.get("name", "?") for d in self.meta["docs"]}
        return self._matrix, self._rows, dict(self._texts), names

    def invalidate(self) -> None:
        self._matrix, self._rows = None, []
        self._texts.clear()

    def save(self) -> None:
        _write_json(self.root / "kb.json", self.meta)

    def remove_files(self, doc_id: str) -> None:
        for ext in (".json", ".npy"):
            try:
                (self.root / "docs" / f"{doc_id}{ext}").unlink(missing_ok=True)
            except OSError:
                pass  # stale file; unreferenced by kb.json


class KBStore:
    """Every knowledge base under one root directory. Mutations and index
    builds hold `lock`; embedding runs outside it."""

    def __init__(self, root: str | Path = DEFAULT_KB_DIR) -> None:
        self.root = Path(root)
        self.lock = threading.RLock()
        self._kbs: dict[str, KnowledgeBase] | None = None

    def _all(self) -> dict[str, KnowledgeBase]:
        if self._kbs is None:
            self._kbs = {}
            if self.root.is_dir():
                for d in sorted(self.root.iterdir()):
                    try:
                        meta = json.loads((d / "kb.json").read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue  # not a knowledge base directory
                    if meta.get("id") == d.name:
                        self._kbs[d.name] = KnowledgeBase(d, meta)
        return self._kbs

    def list(self) -> list[dict]:
        with self.lock:
            return sorted((kb.summary() for kb in self._all().values()),
                          key=lambda s: s["created"])

    def get(self, kb_id: str) -> KnowledgeBase:
        with self.lock:
            kb = self._all().get(str(kb_id))
        if kb is None:
            raise KBError(f"unknown knowledge base: {kb_id}", 404)
        return kb

    def create(self, name: str, embedder: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> dict:
        from .registry import detect_kind
        name = (name or "").strip()
        if not 1 <= len(name) <= 80:
            raise KBError("name must be 1-80 characters")
        if not embedder or not Path(embedder).is_dir() or detect_kind(Path(embedder)) != "embed":
            raise KBError(f"not an embedding model directory: {embedder}")
        if not 100 <= chunk_size <= 8000:
            raise KBError("chunk size must be 100-8000 characters")
        if not 0 <= chunk_overlap <= chunk_size // 4:
            raise KBError(f"overlap must be 0-{chunk_size // 4} (a quarter of the chunk size)")
        with self.lock:
            if any(kb.name.lower() == name.lower() for kb in self._all().values()):
                raise KBError(f"a knowledge base named {name!r} already exists", 409)
            kb_id = _new_id()
            root = self.root / kb_id
            (root / "docs").mkdir(parents=True)
            kb = KnowledgeBase(root, {
                "id": kb_id, "name": name, "created": time.time(),
                "embedder": embedder_profile(embedder), "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap), "docs": []})
            kb.save()
            self._all()[kb_id] = kb
            return kb.summary()

    def delete(self, kb_id: str) -> None:
        with self.lock:
            kb = self.get(kb_id)
            if not _ID_RE.match(kb.id):  # never rmtree outside a KB directory
                raise KBError(f"refusing to delete {kb.root}", 500)
            try:
                shutil.rmtree(kb.root)
            except OSError as e:
                raise KBError(f"could not delete {kb.root}: {e}", 500) from None
            del self._all()[kb.id]

    def add_document(self, kb_id: str, *, name: str, chunks: list[str],
                     vectors: np.ndarray, source: str, sha1: str, chars: int,
                     path: str | None = None) -> dict:
        """Store a document; one with the same name is replaced."""
        with self.lock:
            kb = self.get(kb_id)
            emb = kb.meta["embedder"]
            dim = int(vectors.shape[1])
            if emb.get("dim") and emb["dim"] != dim:
                raise KBError(f"embedding dim {dim} does not match the knowledge "
                              f"base ({emb['dim']})", 409)
            emb["dim"] = dim
            doc = {"id": _new_id(), "name": name, "source": source, "path": path,
                   "sha1": sha1, "chunker": CHUNKER_VERSION, "chars": chars,
                   "chunks": len(chunks), "added": time.time()}
            _write_json(kb.root / "docs" / f"{doc['id']}.json",
                        {"name": name, "chunks": chunks})
            _write_npy(kb.root / "docs" / f"{doc['id']}.npy", vectors.astype(np.float32))
            old = kb.doc_by_name(name)
            kb.meta["docs"] = [d for d in kb.meta["docs"] if d is not old] + [doc]
            kb.save()
            if old:
                kb.remove_files(old["id"])
            kb.invalidate()
            return doc

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        with self.lock:
            kb = self.get(kb_id)
            if kb.doc(doc_id) is None:
                raise KBError(f"unknown document: {doc_id}", 404)
            kb.meta["docs"] = [d for d in kb.meta["docs"] if d["id"] != doc_id]
            kb.save()
            kb.remove_files(doc_id)
            kb.invalidate()


# --------------------------------------------------------------------------- #
# resident embedding / reranking pipelines
# --------------------------------------------------------------------------- #

class Retriever:
    """Embedding and reranking pipelines kept resident beside the chat model.

    Up to MAX_EMBEDDERS embedders stay loaded (knowledge bases may be built
    with different ones) plus one reranker, each keyed by (path, device) and
    guarded by its own inference lock, so an ingestion batch and a chat
    query contend only when they need the same pipeline."""

    MAX_EMBEDDERS = 2

    def __init__(self) -> None:
        self._lock = threading.Lock()  # the model table and loads
        self._embedders: OrderedDict[tuple, dict] = OrderedDict()
        self._reranker: dict | None = None
        self.loading: str | None = None

    def is_loaded(self, kind: str, path: str, device: str) -> bool:
        key = (str(path), device.upper())
        if kind == "embed":
            return any((e["path"], e["device"]) == key
                       for e in list(self._embedders.values()))
        r = self._reranker
        return r is not None and (r["path"], r["device"]) == key

    def _load(self, kind: str, path: str, device: str, pooling: str | None = None) -> dict:
        import openvino_genai as ovgenai

        from .registry import check
        if not Path(path).is_dir():
            raise KBError(f"{kind} model not found: {path}", 404)
        # same registry gate as every other load (e.g. rejects NPU for the
        # verified retrieval models)
        ns = argparse.Namespace(devices=None, image=None, max_new_tokens=None,
                                min_response_len=None)
        errors = [i.message for i in check(kind, path, device, ns) if i.level == "error"]
        if errors:
            raise KBError("registry: " + " | ".join(errors))
        self.loading = f"{kind} {_model_name(path)} on {device}"
        t0 = time.time()
        try:
            if kind == "embed":
                cfg = ovgenai.TextEmbeddingPipeline.Config()
                cfg.pooling_type = _pooling_enum(ovgenai, pooling)
                cfg.normalize = True
                pipe = ovgenai.TextEmbeddingPipeline(path, device, cfg)
            else:
                cfg = ovgenai.TextRerankPipeline.Config()
                cfg.top_n = 1000  # fixed at construction; return every candidate's score
                pipe = ovgenai.TextRerankPipeline(path, device, cfg)
        except Exception as e:  # noqa: BLE001 - surfaced to the UI
            raise KBError(f"loading {kind} model {_model_name(path)} on {device} "
                          f"failed: {e}", 500) from None
        finally:
            self.loading = None
        return {"pipe": pipe, "path": path, "name": _model_name(path),
                "device": device, "pooling": pooling, "qwen3": is_qwen3(path),
                "lock": threading.Lock(), "load_s": round(time.time() - t0, 1)}

    def _embedder(self, profile: dict, device: str) -> dict:
        key = (profile["path"], device.upper(), profile["pooling"])
        with self._lock:
            entry = self._embedders.get(key)
            if entry is not None:
                self._embedders.move_to_end(key)
                return entry
        # compile outside the table lock so a miss does not stall a hot
        # embedder or an in-flight ingest batch
        entry = self._load("embed", *key)
        with self._lock:
            existing = self._embedders.get(key)
            if existing is not None:
                self._embedders.move_to_end(key)
                return existing
            self._embedders[key] = entry
            while len(self._embedders) > self.MAX_EMBEDDERS:
                self._embedders.popitem(last=False)
            return entry

    def _rerank_entry(self, path: str, device: str) -> dict:
        device = device.upper()
        with self._lock:
            r = self._reranker
            if r is not None and (r["path"], r["device"]) == (path, device):
                return r
            self._reranker = None  # drop the old pipeline before compiling
        entry = self._load("rerank", path, device)
        with self._lock:
            r = self._reranker
            if r is not None and (r["path"], r["device"]) == (path, device):
                return r
            self._reranker = entry
            return entry

    def ensure(self, kind: str, path: str, device: str, pooling: str | None = None) -> None:
        """Load a pipeline now rather than on first use."""
        if kind == "embed":
            self._embedder({"path": path, "pooling": pooling}, device)
        else:
            self._rerank_entry(path, device)

    def embed(self, profile: dict, device: str, texts: list[str], *,
              progress=None, cancelled=None) -> np.ndarray:
        """Unit-length embeddings [len(texts), dim], computed in batches;
        `progress(done, total)` runs after each batch, `cancelled()` before."""
        entry = self._embedder(profile, device)
        out: list = []
        for i in range(0, len(texts), EMBED_BATCH):
            if cancelled is not None and cancelled():
                raise KBError("cancelled", 409)
            with entry["lock"]:
                out.extend(entry["pipe"].embed_documents(texts[i:i + EMBED_BATCH]))
            if progress is not None:
                progress(min(i + EMBED_BATCH, len(texts)), len(texts))
        m = np.asarray(out, dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def rerank(self, path: str, device: str, query: str, texts: list[str]) -> list[float]:
        """Relevance score per text (same order), from a cross-encoder."""
        entry = self._rerank_entry(path, device)
        # GenAI feeds Qwen3-Reranker its inputs verbatim; without the official
        # yes/no template its scores are near-random
        q, docs = qwen3_rerank_wrap(query, texts, None) if entry["qwen3"] else (query, texts)
        with entry["lock"]:
            ranked = entry["pipe"].rerank(q, docs)
        scores = [0.0] * len(texts)
        for i, s in ranked:
            scores[i] = float(s)
        return scores

    def describe(self) -> dict:
        def row(e: dict) -> dict:
            return {k: e[k] for k in ("name", "path", "device", "load_s")}
        return {"embedders": [row(e) for e in reversed(list(self._embedders.values()))],
                "reranker": row(self._reranker) if self._reranker else None,
                "loading": self.loading}

    def unload(self) -> None:
        with self._lock:
            self._embedders.clear()
            self._reranker = None


# --------------------------------------------------------------------------- #
# retrieval + prompt
# --------------------------------------------------------------------------- #

def search(store: KBStore, retriever: Retriever, kb_id: str, query: str, *,
           top_k: int = 20, top_n: int = 4, min_score: float = 0.0,
           reranker: str | None = None, embed_device: str = "CPU",
           rerank_device: str = "CPU", on_stage=None) -> dict:
    """Passages for `query`: cosine top-k over every chunk, optionally
    re-scored by a cross-encoder, then the top-n whose score (rerank score
    when reranked, else cosine) is at least `min_score`. Returns the kept
    `hits` and every scored candidate (for inspection)."""
    stage = on_stage or (lambda name, detail=None: None)
    kb = store.get(kb_id)
    emb = kb.meta["embedder"]
    if not Path(emb["path"]).is_dir():
        raise KBError(f"embedding model of {kb.name!r} not found: {emb['path']}", 409)
    with store.lock:
        matrix, rows, texts, names = kb.index()
    if not rows:
        raise KBError(f"knowledge base {kb.name!r} has no documents yet", 409)

    timings: dict[str, int] = {}
    t = time.perf_counter()
    if not retriever.is_loaded("embed", emb["path"], embed_device):
        stage("load_embed", emb["name"])
        retriever.ensure("embed", emb["path"], embed_device, emb["pooling"])
    if reranker and not retriever.is_loaded("rerank", reranker, rerank_device):
        stage("load_rerank", _model_name(reranker))
        retriever.ensure("rerank", reranker, rerank_device)
    if time.perf_counter() - t > 0.05:
        timings["load_ms"] = round((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    q = retriever.embed(emb, embed_device, [(emb.get("query_instruction") or "") + query])[0]
    timings["embed_ms"] = round((time.perf_counter() - t) * 1000)
    if q.shape[0] != matrix.shape[1]:
        raise KBError(f"query embedding dim {q.shape[0]} does not match the "
                      f"knowledge base ({matrix.shape[1]})", 409)

    t = time.perf_counter()
    sims = matrix @ q
    k = min(top_k, len(rows))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx], kind="stable")]
    cands = []
    for j in idx:
        doc_id, chunk_i = rows[j]
        chunks = texts.get(doc_id) or []
        if chunk_i >= len(chunks):
            continue  # snapshot older than a replace that dropped this doc
        cands.append({"doc": doc_id, "chunk": chunk_i,
                      "name": names.get(doc_id, "?"), "text": chunks[chunk_i],
                      "cosine": round(float(sims[j]), 4)})
    timings["search_ms"] = round((time.perf_counter() - t) * 1000)

    if reranker:
        stage("rerank", len(cands))
        t = time.perf_counter()
        scores = retriever.rerank(reranker, rerank_device, query, [c["text"] for c in cands])
        timings["rerank_ms"] = round((time.perf_counter() - t) * 1000)
        for c, s in zip(cands, scores):
            c["rerank"] = round(s, 4)
        cands.sort(key=lambda c: -c["rerank"])
    for c in cands:
        c["score"] = c.get("rerank", c["cosine"])
        c["kept"] = False
    hits = [c for c in cands if c["score"] >= min_score][:top_n]
    for h in hits:
        h["kept"] = True
    return {"kb": {"id": kb.id, "name": kb.name}, "query": query,
            "embedder": emb["name"], "reranker": _model_name(reranker) if reranker else None,
            "hits": hits, "candidates": cands,
            "best": cands[0]["score"] if cands else None, "timings": timings}


_SOFT_SWITCH = re.compile(r"(?<!\S)/(?:no_)?think(?!\S)")


def retrieval_query(question: str) -> str:
    """The question minus chat-control tokens (Qwen3's /think and /no_think
    soft switches), which skew embedding and reranking scores; the prompt
    keeps them so the model still honors them."""
    return _SOFT_SWITCH.sub(" ", question).strip() or question


RAG_PROMPT = {
    "en": ("Answer the question using the reference passages below, retrieved "
           "from the knowledge base \"{kb}\". Cite the passages you rely on by "
           "number, like [1]. If they do not contain the answer, say so, then "
           "answer from your own knowledge if you can. Reply in the language "
           "of the question.", "Question: "),
    "zh": ("请参考下面从知识库「{kb}」中检索到的资料回答问题，并用编号标注所依据的资料，"
           "例如 [1]。如果资料中没有答案，请先说明，再视情况根据自己的知识作答。", "问题："),
}
_CJK = re.compile(r"[㐀-鿿豈-﫿]")


def augment(question: str, hits: list[dict], kb_name: str) -> str:
    """The user turn with retrieved passages spliced in, numbered for
    citation; the instructions follow the question's language (small
    models follow same-language instructions more reliably)."""
    head, label = RAG_PROMPT["zh" if _CJK.search(question) else "en"]
    passages = "\n\n".join(f"[{i}] {h['name']}\n{h['text']}"
                           for i, h in enumerate(hits, 1))
    return f"{head.format(kb=kb_name)}\n\n{passages}\n\n{label}{question}"


# --------------------------------------------------------------------------- #
# ingestion
# --------------------------------------------------------------------------- #

_SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "venv"}


def collect_files(path: str) -> list[dict]:
    """Ingestion items for a local file, or for every supported file under a
    folder (recursive; hidden and dependency folders skipped). Names are
    relative to the folder's parent, so adding the folder again refreshes
    changed files and skips unchanged ones."""
    p = Path(path).expanduser()
    if p.is_file():
        return [{"name": p.name, "path": str(p.resolve()), "source": "path"}]
    if not p.is_dir():
        raise KBError(f"not a file or folder: {path}")
    items: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d not in _SKIP_DIRS)
        for fname in sorted(filenames):
            full = Path(dirpath) / fname
            if full.suffix.lower() not in SUPPORTED_EXTS:
                continue
            items.append({"name": (Path(p.name) / full.relative_to(p)).as_posix(),
                          "path": str(full.resolve()), "source": "path"})
            if len(items) > MAX_FOLDER_FILES:
                raise KBError(f"more than {MAX_FOLDER_FILES} files under {path}")
    if not items:
        raise KBError(f"no supported files under {path}")
    return items


class Ingestor:
    """Background ingestion: one worker drains a FIFO of documents (extract
    text, chunk, embed in batches, store) and publishes progress for the UI
    to poll. Re-adding a document name replaces the stored copy; identical
    content (same SHA-1) is skipped without re-embedding."""

    def __init__(self, store: KBStore, retriever: Retriever) -> None:
        self.store, self.retriever = store, retriever
        self._cv = threading.Condition()
        self._queue: deque[dict] = deque()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self.current: dict | None = None
        self.recent: deque[dict] = deque(maxlen=50)

    def submit(self, kb_id: str, items: list[dict], device: str) -> int:
        self.store.get(kb_id)  # 404 before anything is queued
        with self._cv:
            self._queue.extend({**it, "kb": kb_id, "device": device} for it in items)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="kb-ingest",
                                                daemon=True)
                self._worker.start()
            self._cv.notify()
        return len(items)

    def cancel(self) -> int:
        """Drop queued documents and stop the one in progress."""
        with self._cv:
            dropped = len(self._queue)
            self._queue.clear()
            if self.current is not None:
                self._cancel.set()
        return dropped

    def status(self) -> dict:
        with self._cv:
            return {"active": self.current is not None,
                    "current": dict(self.current) if self.current else None,
                    "queued": len(self._queue),
                    "recent": list(reversed(self.recent))}

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                item = self._queue.popleft()
                self._cancel.clear()
                self.current = {"kb": item["kb"], "name": item["name"],
                                "stage": "read", "done": 0, "total": 0}
            result = {"kb": item["kb"], "name": item["name"]}
            try:
                result.update(self._ingest(item, self.current))
            except Exception as e:  # noqa: BLE001 - reported per document
                result["error"] = str(e)
            result["at"] = time.time()
            with self._cv:
                self.recent.append(result)
                self.current = None

    def _ingest(self, item: dict, cur: dict) -> dict:
        kb = self.store.get(item["kb"])
        if "text" in item:  # pasted text
            raw = item["text"].encode("utf-8")
        elif "data" in item:  # uploaded file
            raw = item["data"]
        else:  # local path
            path = Path(item["path"])
            if path.stat().st_size > MAX_FILE_BYTES:
                raise KBError(f"file larger than {MAX_FILE_BYTES >> 20} MB")
            raw = path.read_bytes()
        sha1 = hashlib.sha1(raw).hexdigest()
        with self.store.lock:
            old = kb.doc_by_name(item["name"])
        if old is not None and old.get("sha1") == sha1 \
                and old.get("chunker") == CHUNKER_VERSION:
            return {"skipped": True, "chunks": old["chunks"]}

        text = normalize_text(item["text"]) if "text" in item else \
            extract_text(item["name"], raw)
        if len(text) > MAX_DOC_CHARS:
            raise KBError(f"document longer than {MAX_DOC_CHARS:,} characters")
        cur["stage"] = "chunk"
        chunks = chunk_text(text, kb.meta["chunk_size"], kb.meta["chunk_overlap"])
        if not chunks:
            raise KBError("no extractable text (a scanned PDF or an empty file?)")
        emb = kb.meta["embedder"]
        if not Path(emb["path"]).is_dir():
            raise KBError(f"embedding model not found: {emb['path']}", 409)
        loaded = self.retriever.is_loaded("embed", emb["path"], item["device"])
        cur.update(stage="embed" if loaded else "load", total=len(chunks))

        def progress(done: int, total: int) -> None:
            cur.update(stage="embed", done=done)

        prefix = emb.get("doc_instruction") or ""
        vectors = self.retriever.embed(emb, item["device"], [prefix + c for c in chunks],
                                       progress=progress, cancelled=self._cancel.is_set)
        cur["stage"] = "save"
        doc = self.store.add_document(
            kb.id, name=item["name"], chunks=chunks, vectors=vectors,
            source=item.get("source", "upload"), path=item.get("path"),
            sha1=sha1, chars=len(text))
        return {"chunks": doc["chunks"], "replaced": old is not None}
