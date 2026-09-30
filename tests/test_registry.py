"""Registry gating: quant detection, model-dir classification, (model x
device x params) rules and local discovery."""
from __future__ import annotations

import argparse
import json

import pytest

from ovtool.registry import (check, detect_kind, detect_quant, find_entry,
                             find_local_models, load, normalize_device)
from tests.conftest import make_model


def ns(**kw):
    base = dict(devices=None, image=None, max_new_tokens=None, min_response_len=None)
    base.update(kw)
    return argparse.Namespace(**base)


def errors(issues):
    return [i.message for i in issues if i.level == "error"]


# ---------------- device normalization & quant detection ---------------- #

@pytest.mark.parametrize("raw,want", [
    ("NPU.0", "NPU"), ("npu", "NPU"), ("HETERO:NPU,GPU", "HETERO"),
    ("gpu", "GPU"), ("auto", "AUTO"), ("MULTI:GPU,CPU.1", "MULTI"),
])
def test_normalize_device(raw, want):
    assert normalize_device(raw) == want


@pytest.mark.parametrize("path,want", [
    ("models/llm/Qwen3-0.6B/fp16", "fp16"),
    ("x/speecht5-fp16", "fp16"),            # substring match too
    ("models/tts/speecht5-fp32", "fp32"),
    ("models/llm/Qwen3-0.6B/int8", "int8"),
    ("qwen3-06b-awq", "int4-awq-g128"),
    ("sd15-int4-asym-g128", "int4-asym-g128"),  # "asym" must win over "sym"
    ("qwen3-06b-int4-sym", "int4-sym-g128"),
    ("ssd1b-int4-g64", "int4-g64"),
    ("plain-old-int4", "int4-asym-g128"),   # the default ladder bottom
    ("models/llm/Qwen3-0.6B/int4-sym-g128", "int4-sym-g128"),
])
def test_detect_quant(path, want):
    assert detect_quant(path) == want


def test_registry_data_is_self_consistent():
    """Every variant key must round-trip through detect_quant, or the gate
    would warn about the registry's own recommended convert outputs."""
    data = load()
    ids = set()
    for entry in data["models"]:
        assert entry["id"] and entry["kind"]
        assert entry["id"] not in ids, f"duplicate id {entry['id']}"
        ids.add(entry["id"])
        for name, v in entry.get("variants", {}).items():
            assert detect_quant(name) == name, (entry["id"], name)
            devices = v.get("devices")
            assert devices, (entry["id"], name)
            assert all(d == d.upper() for d in devices), (entry["id"], name)


def test_find_entry_matches_and_ordering():
    assert find_entry("models/vlm/gemma-4-E2B-it/int8")["id"] == "google/gemma-4-E2B-it"
    assert find_entry("models/embed/Qwen3-Embedding-0.6B/int8")["id"] == \
        "Qwen/Qwen3-Embedding-0.6B"
    assert find_entry("models/rerank/bge-reranker-v2-m3/fp16")["id"] == \
        "BAAI/bge-reranker-v2-m3"
    assert find_entry("totally/unknown-model") is None


# ---------------- detect_kind on fabricated directories ---------------- #

@pytest.mark.parametrize("kind,want", [
    ("llm", "llm"), ("vlm", "vlm"), ("image", "image"),
    ("tts", "tts"),          # SpeechT5: postnet IR
    ("kokoro", "tts"),       # Kokoro: single IR + model_type
    ("embed", "embed"), ("rerank", "rerank"),
])
def test_detect_kind(tmp_path, kind, want):
    assert detect_kind(make_model(tmp_path / kind, kind)) == want


def test_detect_kind_kokoro_by_voices_dir(tmp_path):
    d = make_model(tmp_path / "voices", "llm")  # single IR + language model layout
    (d / "voices").mkdir()
    (d / "voices" / "af_heart.bin").write_bytes(b"\0" * 4)
    assert detect_kind(d) == "tts"


def test_detect_kind_qwen3_embed_and_rerank_by_path_hint(tmp_path):
    """Qwen3-Embedding/-Reranker export as Qwen3ForCausalLM (LLM layout);
    only the catalog path tells them apart."""
    emb = make_model(tmp_path / "embed" / "Qwen3-Embedding-0.6B" / "int8", "llm",
                     config={"model_type": "qwen3"})
    rer = make_model(tmp_path / "rerank" / "Qwen3-Reranker-0.6B" / "int4", "llm",
                     config={"model_type": "qwen3"})
    assert detect_kind(emb) == "embed"
    assert detect_kind(rer) == "rerank"


def test_detect_kind_empty_dir(tmp_path):
    assert detect_kind(tmp_path) is None


# ---------------- check(): the (model x device x params) gate ---------------- #

QWEN_SYM = "models/llm/Qwen3-0.6B/int4-sym-g128"


def test_llm_sym_on_npu_is_ok():
    issues = check("llm", QWEN_SYM, "NPU", ns())
    assert not errors(issues)


def test_llm_asym_on_npu_requires_sym():
    errs = errors(check("llm", "models/llm/Qwen3-0.6B/int4", "NPU", ns()))
    assert any("symmetric" in m for m in errs)


def test_unknown_model_skips_checks():
    issues = check("llm", "models/llm/Unknown-9B/int4-sym", "NPU", ns())
    assert not errors(issues)
    assert any("not in built-in registry" in i.message for i in issues)


def test_image_whole_pipeline_on_npu_blocked():
    errs = errors(check("image", "models/image/SD-Turbo/int8", "NPU", ns()))
    assert any("segmented" in m for m in errs)


def test_image_segmented_on_npu_allowed():
    issues = check("image", "models/image/SD-Turbo/int8", "NPU",
                   ns(devices="NPU,NPU,GPU"))
    assert not errors(issues)


def test_image_segmented_vae_on_npu_warns():
    issues = check("image", "models/image/SD-Turbo/int8", "NPU",
                   ns(devices="NPU,NPU,NPU"))
    assert not errors(issues)
    assert any("VAE" in i.message for i in issues if i.level == "warn")


def test_tts_on_npu_blocked():
    errs = errors(check("tts", "models/tts/speecht5-fp16", "NPU", ns()))
    assert errs


def test_text_only_vlm_rejects_images():
    errs = errors(check("vlm", "models/vlm/Qwen3-VL-2B/int4", "GPU",
                        ns(image=["a.png"])))
    assert any("image input" in m for m in errs)


def test_text_only_vlm_text_prompt_ok():
    assert not errors(check("vlm", "models/vlm/Qwen3-VL-2B/int4", "GPU", ns()))


def test_unverified_device_for_variant():
    errs = errors(check("vlm", "models/vlm/gemma-4-E2B-it/int8", "NPU", ns()))
    assert any("not a verified combination" in m for m in errs)


def test_unregistered_variant_warns():
    issues = check("image", "models/image/SD-Turbo/fp16", "GPU", ns())
    assert not errors(issues)
    assert any("not in the registry" in i.message for i in issues if i.level == "warn")


def test_auto_meta_device_passes():
    assert not errors(check("llm", "models/llm/Qwen3-0.6B/int4", "AUTO", ns()))


def test_llm_npu_max_new_tokens_over_budget_warns():
    issues = check("llm", QWEN_SYM, "NPU", ns(max_new_tokens=512))
    assert not errors(issues)
    assert any("exceeds the NPU static budget" in i.message
               for i in issues if i.level == "warn")


# ---------------- local discovery ---------------- #

def test_find_local_models_groups_variants(tmp_path):
    make_model(tmp_path / "llm" / "Qwen3-0.6B" / "int4", "llm")
    make_model(tmp_path / "llm" / "Qwen3-0.6B" / "int8", "llm")
    make_model(tmp_path / "image" / "SD-Turbo" / "int8", "image")
    groups = find_local_models([str(tmp_path)], with_size=False)
    by_name = {g["name"]: g for g in groups}
    assert by_name["Qwen3-0.6B"]["kind"] == "llm"
    assert {v["name"] for v in by_name["Qwen3-0.6B"]["variants"]} == {"int4", "int8"}
    assert by_name["SD-Turbo"]["kind"] == "image"


def test_find_local_models_skips_cache_dirs(tmp_path):
    make_model(tmp_path / "llm" / "M" / "int4", "llm")
    cache = tmp_path / "llm" / "M" / "int4" / "cache"
    cache.mkdir()
    (cache / "openvino_language_model.xml").write_text("<net/>")
    groups = find_local_models([str(tmp_path)], with_size=False)
    assert len(groups) == 1  # the cache dir is not reported as its own model
