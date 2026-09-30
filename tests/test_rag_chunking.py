"""RAG chunking: the recursive-boundary splitter and text normalizer."""
from __future__ import annotations

import random

from ovtool.rag import chunk_text, normalize_text


def test_normalize_text_unifies_newlines_and_drops_noise():
    messy = "\ufeffTitle\r\nline  \r\n\n\n\nend\x00"
    assert normalize_text(messy) == "Title\nline\n\nend"


def test_normalize_text_collapses_form_feeds_and_blank_runs():
    assert normalize_text("a\x0cb\n\n\n\nc") == "a\n\nb\n\nc"


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_empty_and_whitespace_only_texts_yield_nothing():
    assert chunk_text("") == []
    assert chunk_text("  \n\n  ") == []


def test_exact_size_boundary_is_not_split():
    text = "A" * 800
    assert chunk_text(text, size=800, overlap=0) == [text]


def test_two_paragraphs_cut_at_boundary():
    text = "A" * 400 + "\n\n" + "B" * 400
    chunks = chunk_text(text, size=800, overlap=0)
    assert chunks == ["A" * 400, "B" * 400]


def test_overlap_snaps_to_boundary_not_raw_chars():
    text = "A" * 400 + "\n\n" + "B" * 400
    chunks = chunk_text(text, size=800, overlap=100)
    # the resumed start lands after the paragraph break, so the second chunk
    # does not carry a partial tail of the first
    assert chunks == ["A" * 400, "B" * 400]


def test_cjk_sentences_cut_at_punctuation():
    text = "天气真好。" * 4
    chunks = chunk_text(text, size=12, overlap=0)
    assert all(len(c) <= 12 for c in chunks)
    assert all(c.endswith("。") for c in chunks)


def test_new_markdown_section_starts_clean_without_overlap():
    text = "A" * 450 + "\n\n# Head\n" + "B" * 200
    chunks = chunk_text(text, size=500, overlap=100)
    assert chunks[0] == "A" * 450
    assert chunks[1].startswith("# Head")   # heading opens the chunk untouched


def test_no_boundary_text_hard_cuts_and_terminates():
    chunks = chunk_text("x" * 2500, size=800, overlap=200)
    assert [len(c) for c in chunks] == [800, 800, 800, 700]


def test_all_chunks_respect_size_on_random_text():
    rng = random.Random(42)
    words = ["alpha", "beta", "gamma", "delta", " ", "\n", "\n\n", "."]
    text = "".join(rng.choice(words) for _ in range(3000))
    chunks = chunk_text(text, size=300, overlap=50)
    assert chunks
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.strip() for c in chunks)
    assert chunks[-1].rstrip()[-10:] == text.rstrip()[-10:]


def test_headings_preferentially_stay_with_their_section():
    text = "# One\n" + "a" * 300 + "\n\n# Two\n" + "b" * 300
    chunks = chunk_text(text, size=350, overlap=50)
    two = [c for c in chunks if c.startswith("# Two")]
    assert len(two) == 1
    assert two[0].endswith("b" * 300)
