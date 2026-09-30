"""Document text extraction (txt/html/docx/pdf dispatch), encoding sniffing
and the retrieval prompt helpers."""
from __future__ import annotations

import io
import zipfile

import pytest

from ovtool.rag import (KBError, _decode, _docx_text, _html_text, augment,
                        extract_text, retrieval_query)


# ---------------- decoding ---------------- #

def test_decode_utf16_by_bom():
    assert _decode(b"\xff\xfe" + "hi".encode("utf-16-le")) == "hi"


def test_decode_utf8_with_bom():
    assert _decode("café".encode("utf-8-sig")) == "café"


def test_decode_gb18030_fallback():
    assert _decode("中文文档".encode("gb18030")) == "中文文档"


def test_decode_latin1_last_resort():
    assert _decode(b"caf\xe9") == "café"


# ---------------- HTML ---------------- #

def test_html_text_drops_scripts_and_breaks_blocks():
    markup = ("<html><head><script>var x=1;</script><style>.a{}</style></head>"
              "<body><p>Hello   world</p><div>Second</div></body></html>")
    assert "var x" not in _html_text(markup)
    assert "Hello world" in _html_text(markup)
    assert "Second" in _html_text(markup)


# ---------------- DOCX ---------------- #

def _docx(paras=(), table=None):
    WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;")

    body = "".join(f'<w:p><w:r><w:t>{esc(p)}</w:t></w:r></w:p>' for p in paras)
    if table:
        rows = "".join(
            "<w:tr>" + "".join(
                f"<w:tc><w:p><w:r><w:t>{esc(c)}</w:t></w:r></w:p></w:tc>"
                for c in row) + "</w:tr>"
            for row in table)
        body += f"<w:tbl>{rows}</w:tbl>"
    xml = f'<w:document xmlns:w="{WNS}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_docx_paragraphs_and_table_cells():
    data = _docx(paras=["Hello", "World"], table=[["a", "b"], ["c", "d"]])
    text = _docx_text(data)
    assert text.startswith("Hello\nWorld\n")
    assert "a | b\nc | d" in text


def test_docx_rejects_non_docx_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("other.txt", "nope")
    with pytest.raises(KBError, match="not a readable"):
        _docx_text(buf.getvalue())


# ---------------- PDF (pypdf stubbed; extraction itself is pypdf's job) ---- #

def test_pdf_dispatch(monkeypatch):
    import pypdf

    class FakePage:
        def extract_text(self):
            return "pdf page text"

    class FakeReader:
        def __init__(self, buf):
            self.pages = [FakePage()]
            self.is_encrypted = False

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)
    assert extract_text("doc.pdf", b"") == "pdf page text"


# ---------------- extension dispatch ---------------- #

def test_unknown_extension_text_content_is_accepted():
    assert extract_text("x.weird", "plain bytes".encode()) == "plain bytes"


def test_unknown_extension_binary_content_rejected():
    with pytest.raises(KBError, match="unsupported file type"):
        extract_text("x.exe", b"MZ\x00\x00")


def test_html_file_extracted_via_markup_path(tmp_path):
    data = "<html><body><p>Hi there</p></body></html>".encode()
    assert extract_text("page.html", data) == "Hi there"


# ---------------- retrieval prompts ---------------- #

def test_retrieval_query_strips_think_switches():
    q = retrieval_query("你好 /think 帮我 /no_think 查一下")
    assert "/think" not in q and "/no_think" not in q
    assert "帮我" in q and "查一下" in q


def test_retrieval_query_keeps_normal_slashes():
    assert retrieval_query("a/b/c") == "a/b/c"


def test_retrieval_query_all_switches_falls_back_to_original():
    assert retrieval_query("/no_think") == "/no_think"


def test_augment_chooses_language_and_numbers_hits():
    hits = [{"name": "doc1", "text": "passage one"},
            {"name": "doc2", "text": "passage two"}]
    zh = augment("什么是OpenVINO？", hits, "我的库")
    assert "知识库" in zh and "[1] doc1" in zh and "问题：什么是OpenVINO？" in zh
    en = augment("what is X?", hits, "kb1")
    assert "Answer the question" in en and "[2] doc2" in en
