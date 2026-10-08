"""embed/rerank helpers: pooling-config reading, Qwen3 detection and the
official Qwen3-Reranker template."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ovtool.embed import (QWEN3_QUERY_INSTRUCT, is_multimodal_export, is_qwen3,
                          qwen3_rerank_wrap, read_pooling, run_embed)


def test_run_embed_requires_input():
    with pytest.raises(SystemExit, match="nothing to embed"):
        run_embed(SimpleNamespace(texts=[], query=None, image=None))


def test_is_multimodal_export(tmp_path):
    text_only = tmp_path / "text-embedder"
    text_only.mkdir()
    (text_only / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    vl = tmp_path / "vl-embedder"
    vl.mkdir()
    (vl / "openvino_vision_embeddings_model.xml").write_text("<net/>", encoding="utf-8")
    assert not is_multimodal_export(str(text_only))
    assert is_multimodal_export(str(vl))


def write_pooling(tmp_path, cfg):
    d = tmp_path / "1_Pooling"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path


def test_read_pooling_last_token_spelling_variants(tmp_path):
    assert read_pooling(str(write_pooling(tmp_path, {"pooling_mode_lasttoken": True}))) == "last_token"
    assert read_pooling(str(write_pooling(tmp_path, {"pooling_mode_last_token": True}))) == "last_token"


def test_read_pooling_cls_and_mean(tmp_path):
    assert read_pooling(str(write_pooling(tmp_path, {"pooling_mode_cls_token": True}))) == "cls"
    assert read_pooling(str(write_pooling(tmp_path, {"pooling_mode_mean_tokens": True}))) == "mean"


def test_read_pooling_absent_or_broken(tmp_path):
    assert read_pooling(str(tmp_path)) is None                       # no file
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text("not json", encoding="utf-8")
    assert read_pooling(str(tmp_path)) is None                       # invalid
    assert read_pooling(str(write_pooling(tmp_path, {"pooling_mode_mean_tokens": False,
                                                     "pooling_mode_cls_token": False}))) is None


def test_is_qwen3(tmp_path):
    cases = [("qwen3", True), ("Qwen3", True), ("qwen2", False), ("", False)]
    for i, (mt, want) in enumerate(cases):
        d = tmp_path / f"m{i}"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": mt}), encoding="utf-8")
        assert is_qwen3(str(d)) is want
    (tmp_path / "empty").mkdir()
    assert is_qwen3(str(tmp_path / "empty")) is False


def test_qwen3_rerank_wrap_builds_official_template():
    query, docs = qwen3_rerank_wrap("what is ov?", ["doc A", "doc B"], None)
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert query.startswith("<|im_start|>system")
    assert "Judge whether the Document meets the requirements" in query
    assert QWEN3_QUERY_INSTRUCT in query          # default instruct
    assert query.endswith("<Query>: what is ov?\n")
    assert docs == [f"<Document>: doc A{suffix}", f"<Document>: doc B{suffix}"]


def test_qwen3_rerank_wrap_custom_instruction():
    query, _ = qwen3_rerank_wrap("q", ["d"], "rank these")
    assert "<Instruct>: rank these" in query
