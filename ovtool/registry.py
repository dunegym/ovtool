"""Built-in model compatibility registry.

Loads ovtool/registry.yaml and validates (model x device x parameters)
combinations before a pipeline is loaded, so unusable configurations fail
fast with actionable guidance instead of crashing inside the runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent / "registry.yaml"

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
        if kind in ("llm", "vlm", "image"):
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
