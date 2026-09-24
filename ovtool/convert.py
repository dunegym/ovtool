"""Model conversion & quantization via optimum-intel Python API (no subprocess)."""
from __future__ import annotations

import argparse
from pathlib import Path

TASKS = {
    "llm": "text-generation-with-past",
    "vlm": "image-text-to-text",
    "image": "text-to-image",
}

# int4 format shorthands -> (symmetric, group_size); group_size<=0 means per-channel
INT4_PRESETS = {
    "int4": (False, 128),
    "int4_symg128": (True, 128),
    "int4_symg64": (True, 64),
    "int4_symg32": (True, 32),
    "int4_g128": (False, 128),
    "int4_g64": (False, 64),
    "int4_g32": (False, 32),
}

DEFAULT_SPEECHT5_VOCODER = "microsoft/speecht5_hifigan"
POOLING_LOCAL = "1_Pooling/config.json"


def _hint(module: str) -> SystemExit:
    return SystemExit(
        f"Missing dependency: {module}. Install conversion extras first:\n"
        "  pip install 'optimum-intel[openvino]' onnx"
    )


def _pick_image_cls(model_id: str):
    """Select the right optimum-intel diffusion class from the model config.

    Diffusion repos are diffusers pipelines: their class lives in
    model_index.json (_class_name), not in a transformers AutoConfig.
    """
    name = ""
    try:
        import json
        from huggingface_hub import hf_hub_download
        index_path = hf_hub_download(model_id, "model_index.json")
        with open(index_path, encoding="utf-8") as f:
            name = json.load(f).get("_class_name", "")
    except Exception:
        pass
    if not name:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id)
            name = getattr(cfg, "_class_name", "") or ""
        except Exception:
            pass
    try:
        from optimum.intel import (
            OVFlux2KleinPipeline,
            OVFluxPipeline,
            OVLatentConsistencyModelPipeline,
            OVStableDiffusion3Pipeline,
            OVStableDiffusionPipeline,
            OVStableDiffusionXLPipeline,
        )
    except ImportError as e:
        raise _hint("optimum-intel") from e
    if "Flux2Klein" in name or "Flux2" in name:
        return OVFlux2KleinPipeline
    if "Flux" in name:
        return OVFluxPipeline
    if "StableDiffusion3" in name or "SD3" in name:
        return OVStableDiffusion3Pipeline
    if "StableDiffusionXL" in name or "SDXL" in name:
        return OVStableDiffusionXLPipeline
    if "LatentConsistency" in name:
        return OVLatentConsistencyModelPipeline
    return OVStableDiffusionPipeline


def _pick_lm_cls(kind: str, trust_remote_code: bool):
    try:
        if kind == "vlm":
            from optimum.intel import OVModelForVisualCausalLM as cls
        else:
            from optimum.intel import OVModelForCausalLM as cls
    except ImportError as e:
        raise _hint("optimum-intel") from e
    return cls


def _pick_tts_cls():
    """TTS model class; optimum-intel dispatches SpeechT5 vs Kokoro itself
    from the model config (model_type 'kokoro' / SpeechT5 architecture)."""
    try:
        from optimum.intel import OVModelForTextToSpeechSeq2Seq as cls
    except ImportError as e:
        raise _hint("optimum-intel") from e
    return cls


def _pick_embed_cls():
    try:
        from optimum.intel import OVModelForFeatureExtraction as cls
    except ImportError as e:
        raise _hint("optimum-intel") from e
    return cls


def _pick_rerank_cls():
    try:
        from optimum.intel import OVModelForSequenceClassification as cls
    except ImportError as e:
        raise _hint("optimum-intel") from e
    return cls


def _copy_pooling_config(args: argparse.Namespace, out: Path) -> None:
    """Preserve the sentence-transformers pooling config next to the export.

    openvino-genai does not auto-detect pooling; 'ovtool embed' reads this
    file (pooling_mode_cls / pooling_mode_mean) and applies it."""
    import shutil

    src = Path(args.model) / POOLING_LOCAL if Path(args.model).is_dir() else None
    if src is None or not src.is_file():
        try:
            from huggingface_hub import hf_hub_download
            src = Path(hf_hub_download(args.model, POOLING_LOCAL))
        except Exception:
            src = None
    if src is not None and src.is_file():
        dst = out / POOLING_LOCAL
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Pooling config copied ({POOLING_LOCAL}).")
    else:
        print("No sentence-transformers 1_Pooling config found; 'ovtool embed' "
              "defaults to MEAN pooling (override with --pooling).")


def _compress_fp16_ir(xmls) -> None:
    """Apply OpenVINO fp16 weight compression in place (the same offline
    transformation optimum's .half() applies per model).

    On Windows the read model keeps the original .bin mapped, so the
    compressed model is serialized next to it and moved over the original.
    """
    import gc
    import os

    import openvino as ov
    from openvino._offline_transformations import (apply_moc_transformations,
                                                   compress_model_transformation)
    core = ov.Core()
    for xml in xmls:
        m = core.read_model(str(xml))
        apply_moc_transformations(m, cf=False)
        compress_model_transformation(m)
        tmp = xml.with_name(xml.stem + ".fp16.xml")
        ov.serialize(m, str(tmp))
        del m
        gc.collect()  # release the mapped original .bin before replacing it
        os.replace(tmp, xml)
        os.replace(tmp.with_suffix(".bin"), xml.with_suffix(".bin"))
        print(f"fp16 compression applied: {xml.name}")


def _run_convert_tts(args: argparse.Namespace, out: Path, quant_cfg) -> None:
    """Export a TTS model (SpeechT5 family or Kokoro) to OpenVINO IR.

    optimum-intel's from_pretrained(export=True) drops model_kwargs on the
    way to optimum's exporter, which the SpeechT5 path requires (vocoder);
    route through optimum's main_export directly, then apply the requested
    weight compression on the exported IRs.
    """
    try:
        from optimum.exporters.openvino import main_export
        from optimum.intel import OVConfig
    except ImportError as e:
        raise _hint("optimum-intel") from e

    kokoro = "kokoro" in args.model.lower()
    if kokoro and not args.trust_remote_code:
        raise SystemExit("Kokoro export requires --trust-remote-code (custom "
                         "modeling code from the HF repo); the 'kokoro' pip "
                         "package is also needed at export time")
    try:
        # library inference handles the model_type-less Kokoro config
        from optimum.intel.utils.modeling_utils import \
            _infer_library_from_model_name_or_path
        library_name = _infer_library_from_model_name_or_path(args.model)
    except ImportError:
        library_name = "kokoro" if kokoro else "transformers"
    model_kwargs = None if kokoro else {"vocoder": args.vocoder}
    main_export(model_name_or_path=args.model, output=str(out),
                task="text-to-audio", library_name=library_name,
                trust_remote_code=args.trust_remote_code,
                model_kwargs=model_kwargs,
                ov_config=OVConfig(dtype="fp32"))

    xmls = sorted(out.glob("openvino_*.xml"))
    if not xmls:
        raise SystemExit(f"TTS export produced no OpenVINO IR in {out}")
    if args.weight_format == "fp16":
        _compress_fp16_ir(xmls)
    elif quant_cfg is not None:  # int8 / int4 via optimum-intel (NNCF)
        _pick_tts_cls().from_pretrained(
            str(out), quantization_config=quant_cfg).save_pretrained(str(out))


def _build_quant_cfg(args: argparse.Namespace):
    """Return an OVWeightQuantizationConfig or None (fp32/fp16 passthrough).

    Models saved via OpenVINO serialize with compress_to_fp16=True by default,
    so 'fp16' needs no extra step; fp32 is only meaningful as 'skip quantization'.
    """
    if args.weight_format == "fp32":
        return None
    if args.weight_format == "fp16":
        return None
    try:
        from optimum.intel import OVWeightQuantizationConfig
    except ImportError as e:
        raise _hint("optimum-intel") from e

    if args.weight_format == "int8":
        return OVWeightQuantizationConfig(bits=8)
    sym, group_size = INT4_PRESETS[args.weight_format]
    if args.sym:
        sym = True
    if args.asym:
        sym = False
    if args.group_size is not None:
        group_size = args.group_size
    kwargs = dict(bits=4, sym=sym, group_size=group_size,
                  ratio=args.ratio if args.ratio is not None else 1.0)
    if args.awq:
        # text models calibrate on wikitext2; visual-language models need a
        # multimodal dataset (textvqa) plus an explicit processor for the
        # calibration builder
        default_awq_dataset = "textvqa" if args.kind == "vlm" else "wikitext2"
        kwargs.update(quant_method="awq",
                      dataset=args.dataset or default_awq_dataset)
        if args.kind == "vlm":
            kwargs["processor"] = args.model
    elif args.dataset:
        kwargs.update(dataset=args.dataset)
    return OVWeightQuantizationConfig(**kwargs)


def _save_tokenizer(args: argparse.Namespace, out: Path, subdir: str | None = None,
                    src_subfolder: str | None = None) -> None:
    """Convert & save the HF tokenizer into OpenVINO tokenizer IR + copy configs.

    openvino_genai pipelines (LLM/VLM and diffusion alike) require
    openvino_tokenizer.xml for prompt encoding. Diffusion pipelines expect it
    inside the `tokenizer/` component subfolder (GenAI derives the path from
    text_encoder -> tokenizer); LLM/VLM expect it at the model root. SDXL-class
    pipelines carry a second encoder whose tokenizer GenAI resolves as
    text_encoder_2 -> tokenizer_2, so pass src_subfolder="tokenizer_2".
    """
    target = out / subdir if subdir else out
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise _hint("transformers") from e
    import openvino as ov

    tok = None
    last_err = None
    # prefer the original model repo (a tokenizer saved by tiktoken-backed
    # transformers can produce a broken tokenizer.json that encodes to []),
    # then the export dir layouts
    candidates = [args.model, out / "tokenizer", target, out]
    for src in candidates:
        try:
            if src_subfolder and src == args.model:
                cand = AutoTokenizer.from_pretrained(
                    str(src), subfolder=src_subfolder,
                    trust_remote_code=args.trust_remote_code)
            else:
                cand = AutoTokenizer.from_pretrained(
                    str(src), trust_remote_code=args.trust_remote_code)
            probe = cand.encode("hello")
            if not probe:
                raise ValueError(f"tokenizer from {src} encodes to empty ids (broken)")
            tok = cand
            break
        except Exception as e:
            last_err = e
    if tok is None:
        print(f"Warning: no usable tokenizer found ({last_err}); skipping tokenizer conversion.")
        return
    target.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(target)  # HF tokenizer configs incl. chat template
    try:
        from openvino_tokenizers import convert_tokenizer
        try:
            result = convert_tokenizer(tok, with_detokenizer=True, streaming_detokenizer=True)
        except Exception:
            result = convert_tokenizer(tok, with_detokenizer=True)
        if isinstance(result, tuple):
            tok_model, detok_model = result
            ov.serialize(tok_model, str(target / "openvino_tokenizer.xml"),
                         str(target / "openvino_tokenizer.bin"))
            ov.serialize(detok_model, str(target / "openvino_detokenizer.xml"),
                         str(target / "openvino_detokenizer.bin"))
        else:
            ov.serialize(result, str(target / "openvino_tokenizer.xml"),
                         str(target / "openvino_tokenizer.bin"))
        print("OpenVINO tokenizer saved.")
    except ImportError:
        print("Warning: openvino-tokenizers not installed; "
              "pip install openvino-tokenizers for streaming inference.")
    except Exception as e:
        print(f"Warning: tokenizer conversion failed ({e}); "
              "generation may fall back to HF tokenizer configs.")


def _save_processor(args: argparse.Namespace, out: Path) -> None:
    """VLMs additionally need the image processor / processor configs."""
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        proc.save_pretrained(out)
    except Exception as e:
        print(f"Warning: processor save failed ({e})")


def _relax_stale_version_guards() -> None:
    """optimum-intel 2.1.0 pins stale MAX_TRANSFORMERS_VERSION guards on some
    newer architectures (e.g. the qwen3-vl text decoder is pinned to 5.0 while
    the rest of the stack supports 5.2+), blocking export on perfectly capable
    transformers versions. Relax those pins; the exporter still validates the
    traced model itself."""
    try:
        from optimum.exporters.openvino import model_configs as ov_configs
    except ImportError:
        return
    for name in dir(ov_configs):
        if ("Qwen3" in name or "Qwen2_5_VL" in name) and "Config" in name:
            cls = getattr(ov_configs, name)
            if getattr(cls, "MAX_TRANSFORMERS_VERSION", None):
                try:
                    cls.MAX_TRANSFORMERS_VERSION = None
                except Exception:
                    pass


def run_convert(args: argparse.Namespace) -> None:
    if args.kind in ("vlm",):
        _relax_stale_version_guards()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    quant_cfg = _build_quant_cfg(args)

    print(f"Exporting {args.model!r} ({args.kind}) -> {out} "
          f"[weight-format={args.weight_format}] ...")
    if args.kind == "tts":
        _run_convert_tts(args, out, quant_cfg)
        if "kokoro" in args.model.lower():
            # Kokoro phonemizes via its own G2P (misaki/espeak-ng), no tokenizer
            print("Kokoro export: no tokenizer conversion needed "
                  "(phonemization runs inside GenAI).")
        else:
            _save_tokenizer(args, out)  # SpeechT5 needs openvino_tokenizer.xml
        print(f"\nDone. OpenVINO IR written to: {out.resolve()}")
        print(f"Run inference with: ovtool tts -m {out}")
        return

    load_kwargs = dict(
        export=True,
        quantization_config=quant_cfg,
        trust_remote_code=args.trust_remote_code,
        use_cache=True,
    )
    if quant_cfg is None:
        # CRITICAL: optimum's exporter auto-quantizes every submodel >=1B
        # parameters to int8_asym when no quantization_config is given
        # (_apply_model_size_based_quantization). Disable it for fp16/fp32.
        load_kwargs["load_in_8bit"] = False

    if args.kind == "image":
        cls = _pick_image_cls(args.model)
        load_kwargs.pop("use_cache", None)
    elif args.kind == "embed":
        cls = _pick_embed_cls()
        load_kwargs.pop("use_cache", None)
    elif args.kind == "rerank":
        cls = _pick_rerank_cls()
        load_kwargs.pop("use_cache", None)
    else:
        cls = _pick_lm_cls(args.kind, args.trust_remote_code)

    try:
        model = cls.from_pretrained(args.model, **load_kwargs)
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Export failed: {e}\n"
                         "Tip: try adding --trust-remote-code for custom models, "
                         "or a different --weight-format.") from e

    if quant_cfg is None and args.weight_format == "fp16":
        # load_in_8bit=False saved the exported (bf16) weights uncompressed;
        # apply the FP16 compression transformation explicitly
        model.half()

    # optimum-intel saves the model + configs; we add the converted tokenizer
    model.save_pretrained(out)
    if args.kind == "image":
        _save_tokenizer(args, out, subdir="tokenizer")
        # SDXL carries two text encoders, SD3.x three (CLIP-L + CLIP-G + T5);
        # GenAI maps text_encoder_N -> tokenizer_N, so each needs its own IR
        for extra in sorted(out.glob("tokenizer_*")):
            if extra.is_dir():
                _save_tokenizer(args, out, subdir=extra.name,
                                src_subfolder=extra.name)
    else:
        _save_tokenizer(args, out)
        if args.kind == "embed":
            _copy_pooling_config(args, out)
    if args.kind == "vlm":
        _save_processor(args, out)
    print(f"\nDone. OpenVINO IR written to: {out.resolve()}")
    print("Run inference with: ovtool " +
          {"llm": "chat", "vlm": "vlm", "image": "image",
           "embed": "embed", "rerank": "rerank"}[args.kind] +
          f" -m {out}")


def add_convert_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("convert", help="Convert a HF model to OpenVINO IR with optional quantization",
                       description="Convert/quantize a Hugging Face model with optimum-intel. "
                                   "Examples:\n"
                                   "  ovtool convert llm Qwen/Qwen3-0.6B -o ./qwen3-06b-int4\n"
                                   "  ovtool convert vlm openbmb/MiniCPM-V-2_6 -m ./minicpmv-int4 --weight-format int4 --sym\n"
                                   "  ovtool convert image stabilityai/sd-turbo -m ./sd-turbo-ir --weight-format int8\n"
                                   "  ovtool convert tts microsoft/speecht5_tts -o ./speecht5-fp16\n"
                                   "  ovtool convert tts hexgrad/Kokoro-82M --trust-remote-code -o ./kokoro\n"
                                   "  ovtool convert embed BAAI/bge-small-en-v1.5 -o ./bge-small\n"
                                   "  ovtool convert rerank BAAI/bge-reranker-v2-m3 -o ./bge-reranker",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["llm", "vlm", "image", "tts", "embed", "rerank"],
                   help="Model family")
    p.add_argument("model", help="Hugging Face model id or local path")
    p.add_argument("-o", "--output", default=None, help="Output dir (default: ./<basename>-<wf>)")
    p.add_argument("--weight-format", default=None,
                   choices=["fp32", "fp16", "int8"] + list(INT4_PRESETS),
                   help="Weight compression format (default int4; tts/embed/rerank default to fp16)")
    p.add_argument("--ratio", type=float, default=None, help="int4 compression ratio, 0.1-1.0 (default 1.0)")
    p.add_argument("--group-size", type=int, default=None, help="int4 group size (default 128; -1 = per-channel)")
    p.add_argument("--sym", action="store_true", help="Force symmetric quantization (recommended for NPU)")
    p.add_argument("--asym", action="store_true", help="Force asymmetric quantization (better CPU/GPU accuracy)")
    p.add_argument("--awq", action="store_true", help="Enable AWQ (activation-aware weight quantization)")
    p.add_argument("--dataset", default=None, help="Calibration dataset (default wikitext2 when AWQ on)")
    p.add_argument("--vocoder", default=DEFAULT_SPEECHT5_VOCODER,
                   help="tts (SpeechT5) only: HiFi-GAN vocoder model id "
                        f"(default {DEFAULT_SPEECHT5_VOCODER})")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow custom modeling code from the HF repo")

    def _set_output(args):
        if args.weight_format is None:
            # small quantization-sensitive families keep fp16 by default
            args.weight_format = "fp16" \
                if args.kind in ("tts", "embed", "rerank") else "int4"
        if args.output is None:
            base = args.model.rstrip("/").split("/")[-1]
            args.output = f"./{base}-{args.weight_format}"
    p.set_defaults(func=run_convert, hook=_set_output)
