"""Device discovery and validation helpers."""
from __future__ import annotations

import openvino as ov

# Devices OpenVINO GenAI officially targets (plus AUTO/HETERO/MULTI meta-devices)
META_DEVICES = {"AUTO", "HETERO", "MULTI", "BATCH"}


def get_core() -> ov.Core:
    return ov.Core()


def list_devices() -> list[dict]:
    """Enumerate available devices with useful properties."""
    core = get_core()
    devices = []
    for dev in core.get_available_devices():
        info = {"device": dev}
        for prop in ("FULL_DEVICE_NAME", "DEVICE_TYPE", "DRIVER_VERSION"):
            try:
                info[prop.lower()] = core.get_property(dev, prop)
            except Exception:
                info[prop.lower()] = "?"
        # expand sub-devices, e.g. GPU -> GPU.0, GPU.1
        try:
            subs = core.get_property(dev, "AVAILABLE_DEVICES")
            if subs:
                info["subdevices"] = list(subs)
        except Exception:
            pass
        devices.append(info)
    return devices


def resolve_device(device: str) -> str:
    """Validate the requested device: a concrete device (CPU/GPU/NPU), a
    subdevice (GPU.1), or a meta-device — bare (AUTO) or compound
    (HETERO:GPU,CPU / MULTI:NPU,CPU), whose components are checked too."""
    core = get_core()
    available = core.get_available_devices()
    upper = device.upper()
    if upper in available:
        return device
    if upper.split(".")[0] in available:
        return upper  # subdevice of an available device, e.g. GPU.1
    token, _, rest = upper.partition(":")
    if token in META_DEVICES:
        # compound form: every component must itself resolve
        comps = [c.split(".")[0].strip() for c in rest.split(",")] if rest else []
        unknown = [c for c in comps
                   if c and c not in available and c not in META_DEVICES]
        if unknown:
            raise SystemExit(
                f"Device '{device}' names unknown component(s) "
                f"{', '.join(unknown)}. Available: {', '.join(available)}.")
        return upper
    raise SystemExit(
        f"Device '{device}' is not available. Available: {', '.join(available)} "
        f"(meta-devices like AUTO, HETERO:GPU,CPU or MULTI:GPU,CPU are also "
        f"accepted)."
    )


def print_devices() -> None:
    rows = list_devices()
    if not rows:
        print("No inference devices found.")
        return
    print(f"{'Device':<12} {'Name':<48} {'Driver':<16} Subdevices")
    print("-" * 100)
    for r in rows:
        subs = ", ".join(r.get("subdevices", [])) or "-"
        print(f"{r['device']:<12} {r.get('full_device_name', '?'):<48} "
              f"{r.get('driver_version', '?'):<16} {subs}")
