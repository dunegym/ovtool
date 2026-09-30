"""convert parser defaults (per-kind weight formats, output naming) and the
quantization-config builder (optimum-intel required only for the latter)."""
from __future__ import annotations

import argparse

import pytest

from ovtool.convert import INT4_PRESETS, add_convert_parser


def parse_convert(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    add_convert_parser(sub)
    args = parser.parse_args(["convert"] + argv)
    args.hook(args)
    return args


@pytest.mark.parametrize("kind,default_wf", [
    ("llm", "int4"), ("vlm", "int4"),
    ("tts", "fp16"), ("embed", "fp16"), ("rerank", "fp16"),
])
def test_default_weight_format_per_kind(kind, default_wf):
    args = parse_convert([kind, "Qwen/Qwen3-0.6B"])
    assert args.weight_format == default_wf
    assert args.output == f"./Qwen3-0.6B-{default_wf}"


def test_explicit_output_and_format_respected():
    args = parse_convert(["llm", "Qwen/Qwen3-0.6B", "-o", "out", "--weight-format", "int8"])
    assert args.output == "out" and args.weight_format == "int8"


def test_trailing_slash_repo_name():
    args = parse_convert(["llm", "Qwen/Qwen3-0.6B/"])
    assert args.output == "./Qwen3-0.6B-int4"


def test_every_int4_preset_is_a_valid_choice():
    for preset in INT4_PRESETS:
        args = parse_convert(["llm", "X", "--weight-format", preset])
        assert args.weight_format == preset


# ---------------- _build_quant_cfg (needs optimum-intel) ---------------- #

def build_cfg(args):
    pytest.importorskip("optimum.intel")   # only these tests need the extra
    from ovtool.convert import _build_quant_cfg
    return _build_quant_cfg(args)


def quant_args(**kw):
    base = dict(weight_format="int4", sym=False, asym=False, awq=False,
                dataset=None, ratio=None, group_size=None, kind="llm",
                model="Qwen/Qwen3-0.6B")
    base.update(kw)
    return argparse.Namespace(**base)


def test_fp32_and_fp16_skip_quantization():
    assert build_cfg(quant_args(weight_format="fp32")) is None
    assert build_cfg(quant_args(weight_format="fp16")) is None


def test_int8_config():
    assert build_cfg(quant_args(weight_format="int8")).bits == 8


def test_int4_preset_and_overrides():
    cfg = build_cfg(quant_args())
    assert (cfg.bits, cfg.sym, cfg.group_size, cfg.ratio) == (4, False, 128, 1.0)
    cfg = build_cfg(quant_args(weight_format="int4_symg64"))
    assert (cfg.sym, cfg.group_size) == (True, 64)
    assert build_cfg(quant_args(sym=True)).sym is True
    assert build_cfg(quant_args(weight_format="int4_symg128", asym=True)).sym is False
    assert build_cfg(quant_args(group_size=64)).group_size == 64
    assert build_cfg(quant_args(ratio=0.5)).ratio == 0.5


def test_awq_calibration_datasets():
    cfg = build_cfg(quant_args(awq=True))
    assert cfg.quant_method == "awq" and cfg.dataset == "wikitext2"
    cfg = build_cfg(quant_args(awq=True, kind="vlm", model="X/Y"))
    assert cfg.dataset == "textvqa" and cfg.processor == "X/Y"
    cfg = build_cfg(quant_args(awq=True, dataset="c4"))
    assert cfg.dataset == "c4"
