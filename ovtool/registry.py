"""Built-in model compatibility registry.

Loads ovtool/registry.yaml and validates (model x device x parameters)
combinations before a pipeline is loaded, so unusable configurations fail
fast with actionable guidance instead of crashing inside the runtime.

Also hosts local-model discovery: recursive scanning of the default models
directory plus any extra roots listed in $OVTOOL_MODELS_PATH (os.pathsep
separated), shared by the `ovtool models local` command and the web UI.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent / "registry.yaml"
MODELS_PATH_ENV = "OVTOOL_MODELS_PATH"
DEFAULT_MODELS_DIR = "./models"

OK, WARN, ERROR = "ok", "warn", "error"


@dataclass
class Issue:
    level: str  # OK | WARN | ERROR
    message: str


def load() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_device(device: str) -> str:
    # "NPU.0" / "npu" / "HETERO:NPU,GPU" -> leading device token
    return device.upper().split(":")[0].split(".")[0].strip()


def detect_quant(model_str: str) -> str:
    import re
    tokens = re.split(r"[\\/_\-. ]", model_str.lower())
    if "fp16" in tokens or "fp16" in model_str.lower():
        return "fp16"
    if "int8" in tokens or "int8" in model_str.lower():
        return "int8"
    if "awq" in tokens:
        return "int4-awq-g128"
    if "asym" in tokens:  # must be tested before "sym" ("asym" contains "sym")
        return "int4-asym-g128"
    if "sym" in tokens:
        return "int4-sym-g128"
    if "g64" in tokens:
        return "int4-g64"
    return "int4-asym-g128"


def find_entry(model_str: str) -> dict | None:
    s = model_str.lower()
    for entry in load().get("models", []):
        if s in entry["id"].lower() or any(m in s for m in entry.get("match", [])):
            return entry
    return None


def check(kind: str, model_str: str, device: str, args: argparse.Namespace) -> list[Issue]:
    """Validate an inference request against the registry."""
    issues: list[Issue] = []
    entry = find_entry(model_str)
    dev = normalize_device(device)

    global_rules = load().get("global_rules", []) if entry else []

    # segmented per-component execution (--devices TE,DENOISE,VAE): text
    # encoder + denoiser may run on the NPU while VAE decode stays on GPU
    seg = [d.strip().upper().split(":")[0].split(".")[0]
           for d in (getattr(args, "devices", "") or "").split(",") if d.strip()]
    segmented = kind == "image" and len(seg) == 3

    if entry is None:
        if kind in ("llm", "vlm", "image", "tts", "embed", "rerank"):
            issues.append(Issue(OK, "model not in built-in registry; skipping compatibility checks"))
        return issues

    quant = detect_quant(model_str)
    variants = entry.get("variants", {})

    # 1) known quantization variants
    if quant not in variants and variants:
        issues.append(Issue(
            WARN,
            f"quantization variant '{quant}' is not in the registry for {entry['id']}; "
            f"known variants: {', '.join(variants)}"))

    # 2) devices allowed for the detected variant (fall back to union);
    #    segmented mode places components individually, so the whole-model
    #    device list does not apply
    allowed: set[str] = set()
    for v in variants.values():
        allowed.update(d.upper() for d in v.get("devices", []))
    if allowed and dev not in allowed and dev not in ("AUTO", "HETERO", "MULTI", "BATCH") \
            and not segmented:
        hints = []
        for name, v in variants.items():
            if dev in [d.upper() for d in v.get("devices", [])]:
                hints.append(f"'{name}' ({', '.join(v['devices'])})")
        msg = f"{entry['id']} on {dev} is not a verified combination for variant '{quant}'."
        if hints:
            msg += f" Verified {dev} variants: {'; '.join(hints)}."
        issues.append(Issue(ERROR, msg))

    # 3) global device/kind/quant rules
    for rule in global_rules:
        if rule.get("when_device") != dev:
            continue
        if rule.get("forbid_kind") == kind:
            if segmented:
                issues.append(Issue(
                    OK, "segmented image execution (--devices) is allowed on NPU: "
                        "text encoder + denoiser on NPU, VAE decode stays on GPU/CPU"))
                continue
            issues.append(Issue(ERROR, f"{kind} inference on NPU is disabled: {rule['message']}"))
        req = rule.get("require_quant")
        kinds = rule.get("for_kinds", [])
        if req and kind in kinds and quant != req:
            issues.append(Issue(ERROR, rule["message"]))

    if segmented and seg[2] == "NPU":
        issues.append(Issue(
            WARN, "VAE decode on NPU is not supported by the runtime; keep the "
                  "third --devices entry on GPU or CPU"))

    # 4) entry-specific rules
    if entry.get("text_only") and kind == "vlm" and getattr(args, "image", None):
        issues.append(Issue(
            ERROR,
            f"image input for {entry['id']} is known-broken on this openvino-genai release "
            "(shape validation error); run text-only prompts for now"))

    if dev == "NPU" and kind == "llm":
        # keep the static response budget consistent with the requested tokens
        min_resp = getattr(args, "min_response_len", None) or 128
        max_new = getattr(args, "max_new_tokens", None)
        if max_new and max_new > min_resp:
            issues.append(Issue(
                WARN,
                f"--max-new-tokens {max_new} exceeds the NPU static budget "
                f"--min-response-len {min_resp}; generation will be capped. "
                "Raise --min-response-len to match."))
        if entry.get("recommended_device") == "GPU" and dev == "NPU":
            pass  # recommendation surfaced below in `models`

    # 5) success note
    if not issues or all(i.level == OK for i in issues):
        rec = entry.get("recommended_device")
        note = f"verified combination: {entry['id']} [{quant}] on {dev}"
        if rec and rec == dev:
            note += " (recommended device)"
        issues.append(Issue(OK, note))
    return issues


def report(issues: list[Issue]) -> bool:
    """Print issues; return False when an ERROR blocks execution."""
    icon = {OK: "[ok]", WARN: "[warn]", ERROR: "[error]"}
    blocked = False
    for i in issues:
        print(f"{icon[i.level]} {i.message}")
        if i.level == ERROR:
            blocked = True
    return not blocked


def print_models(query: str | None = None) -> None:
    data = load()
    for entry in data.get("models", []):
        if query and query.lower() not in entry["id"].lower():
            continue
        variants = entry.get("variants", {})
        devices = sorted({d.upper() for v in variants.values() for d in v.get("devices", [])})
        print(f"{entry['id']}  [{entry['kind']}]")
        print(f"  devices : {', '.join(devices)}")
        for name, v in variants.items():
            note = f" — {v['note']}" if v.get("note") else ""
            print(f"  variant : {name} -> {', '.join(v['devices'])}{note}")
            if v.get("convert"):
                print(f"            {v['convert']}")
        tags = []
        if entry.get("text_only"):
            tags.append("text-only on this GenAI release")
        if entry.get("recommended_device"):
            tags.append(f"recommended: {entry['recommended_device']}")
        if tags:
            print(f"  notes   : {'; '.join(tags)}")
        print()


# --------------------------------------------------------------------------- #
# local model discovery
# --------------------------------------------------------------------------- #

def extra_model_roots() -> list[str]:
    """Extra model roots from $OVTOOL_MODELS_PATH (os.pathsep separated)."""
    raw = os.environ.get(MODELS_PATH_ENV, "")
    return [p.strip().strip('"') for p in raw.split(os.pathsep) if p.strip().strip('"')]


def model_roots(default_dir: str = DEFAULT_MODELS_DIR) -> list[str]:
    return [default_dir] + [r for r in extra_model_roots() if r != default_dir]


def detect_kind(model_dir: Path) -> str | None:
    """Classify a directory as llm/vlm/image/tts from its exported artifacts."""
    try:
        names = {p.name for p in model_dir.iterdir() if p.is_file()}
    except OSError:
        return None
    if "model_index.json" in names:
        return "image"  # diffusers pipeline (component IRs in subfolders)
    if "openvino_vision_embeddings_model.xml" in names or \
            "openvino_text_embeddings_model.xml" in names:
        return "vlm"
    if "openvino_postnet.xml" in names or "openvino_vocoder.xml" in names:
        return "tts"  # SpeechT5 export: encoder/decoder/postnet/vocoder IRs
    if "openvino_language_model.xml" in names or "openvino_model.xml" in names:
        if _config_model_type(model_dir) == "kokoro" or \
                (model_dir / "voices").is_dir():
            return "tts"  # Kokoro export: single IR + voices/*.bin packs
        arch = _config_architectures(model_dir)
        if any("ForSequenceClassification" in a for a in arch):
            return "rerank"  # cross-encoder reranker export
        if arch and all(a.endswith("Model") for a in arch):
            return "embed"  # plain backbone export (BertModel / XLMRobertaModel ...)
        # Qwen3-Embedding / Qwen3-Reranker export as Qwen3ForCausalLM (same IR
        # layout as an LLM); tell them apart from the catalog path convention
        # <root>/<kind>/<model>/<variant> (kind dir or model name hints)
        hints = " ".join((model_dir.parent.name, model_dir.parent.parent.name)).lower()
        if "embed" in hints:
            return "embed"
        if "rerank" in hints:
            return "rerank"
        return "llm"
    return None


def _config_architectures(model_dir: Path) -> list[str]:
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        return [str(a) for a in cfg.get("architectures", [])]
    except (OSError, ValueError):
        return []


def _config_model_type(model_dir: Path) -> str:
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        return str(cfg.get("model_type", "")).lower()
    except (OSError, ValueError):
        return ""


_SKIP_DIRS = {"cache", "__pycache__", ".git", ".mimosa"}


def _dir_size_gb(d: Path) -> float:
    total = 0
    try:
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            if any(part in _SKIP_DIRS for part in f.relative_to(d).parts):
                continue  # e.g. NPU compile caches inside the model dir
            total += f.stat().st_size
    except OSError:
        pass
    return round(total / 1e9, 2)


def scan_root(root: Path) -> list[dict]:
    """Recursively find model directories under root (does not descend into
    a detected model: its component subfolders also contain IR files)."""
    found: list[dict] = []

    def walk(d: Path) -> None:
        kind = detect_kind(d)
        if kind:
            try:
                rel = d.relative_to(root)
            except ValueError:
                rel = d.name
            parts = rel.parts
            found.append({
                "kind": kind,
                "model": parts[-2] if len(parts) >= 2 else root.name,
                "variant": parts[-1] if parts else root.name,
                "path": str(d),
            })
            return
        try:
            subs = sorted(d.iterdir())
        except OSError:
            return
        for sub in subs:
            if sub.is_dir() and sub.name not in _SKIP_DIRS:
                walk(sub)

    walk(root)
    return found


def find_local_models(roots: list[str] | None = None,
                      with_size: bool = True) -> list[dict]:
    """Group locally found models as [{kind, name, root, variants:[...]}].

    Grouping mirrors the <root>/<kind>/<model>/<variant> convention but
    works for any nesting: a detected model directory contributes one
    variant, its parent directory name is the model name.
    """
    roots = model_roots() if roots is None else roots
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []
    for root in roots:
        rootp = Path(root)
        if not rootp.is_dir():
            continue
        for m in scan_root(rootp):
            key = (m["kind"], m["model"], root)
            if key not in groups:
                groups[key] = {"kind": m["kind"], "name": m["model"],
                               "root": root, "variants": []}
                order.append(key)
            entry = {"name": m["variant"], "path": m["path"]}
            if with_size:
                entry["size_gb"] = _dir_size_gb(Path(m["path"]))
            groups[key]["variants"].append(entry)
    return [groups[k] for k in order]


def _local_quants_by_entry() -> dict[str, set[str]]:
    """registry entry id -> set of locally available quant variants."""
    mapping: dict[str, set[str]] = {}
    for group in find_local_models(with_size=False):
        for v in group["variants"]:
            entry = find_entry(v["path"])
            if entry is not None:
                mapping.setdefault(entry["id"], set()).add(detect_quant(v["path"]))
    return mapping


def print_models_remote(query: str | None = None) -> None:
    local = _local_quants_by_entry()
    for entry in load().get("models", []):
        if query and query.lower() not in entry["id"].lower():
            continue
        variants = entry.get("variants", {})
        devices = sorted({d.upper() for v in variants.values() for d in v.get("devices", [])})
        have = local.get(entry["id"], set())
        mark = f"  [local: {len(have)}/{len(variants)} variants]" if variants and have \
            else "  [not local]"
        print(f"{entry['id']}  [{entry['kind']}]  ({', '.join(devices)}){mark}")
        line = ", ".join(f"{name}{' *' if name in have else ''}"
                         for name in variants)
        if line:
            print(f"  variants: {line}   ( * = available locally)")
        tags = []
        if entry.get("text_only"):
            tags.append("text-only on this GenAI release")
        if entry.get("recommended_device"):
            tags.append(f"recommended: {entry['recommended_device']}")
        if tags:
            print(f"  notes   : {'; '.join(tags)}")
        print()


def print_models_local(query: str | None = None) -> None:
    groups = find_local_models()
    roots = sorted({g["root"] for g in groups})
    for r in extra_model_roots():
        if r not in roots:
            print(f"(extra root not found: {r})")
    count = 0
    for g in groups:
        for v in g["variants"]:
            text = f"{g['name']}/{v['name']}"
            if query and query.lower() not in text.lower() \
                    and query.lower() not in v["path"].lower():
                continue
            size = f"  {v.get('size_gb', 0):.2f} GB" if "size_gb" in v else ""
            root_tag = "" if g["root"] == DEFAULT_MODELS_DIR else f"  [{g['root']}]"
            print(f"[{g['kind']}] {text}{size}{root_tag}")
            print(f"        {v['path']}")
            count += 1
    print(f"{count} local model(s)"
          + (f" across {len(roots)} root(s)" if len(roots) > 1 else ""))
