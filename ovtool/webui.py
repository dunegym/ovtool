"""Browser UI for ovtool: chat & image generation on locally converted models.

`ovtool webui` starts an HTTP server that serves a single-page app (webui.html)
plus a small JSON API:

  GET  /                     the web UI
  GET  /api/devices          available OpenVINO devices
  GET  /api/models           models found under --models-dir
  GET  /api/status           currently loaded model / load progress
  POST /api/load             load a model (kind llm|vlm|image) on a device
  POST /api/unload           release the loaded pipeline
  POST /api/chat             chat completion (SSE stream or JSON)
  POST /api/image            text-to-image generation (base64 PNG results)

One pipeline is loaded at a time (loading happens in a background thread;
the UI polls /api/status). Chat requests render the model's chat template
per request (stateless), reusing the serve implementation. Image models may
be loaded in segmented mode ("TE,DENOISE,VAE") which fixes static geometry,
mirroring `ovtool image --devices`.
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import openvino_genai as ovgenai
import openvino_tokenizers  # noqa: F401  (registers custom-op extension)

from .devices import list_devices
from .registry import check as registry_check, report as registry_report
from .server import DEFAULT_MAX_TOKENS, _finish_reason, _usage, build_config, render_chat

WEBUI_HTML = Path(__file__).parent / "webui.html"
KIND_DIRS = ("llm", "vlm", "image")
GENERATION_LOCK_TIMEOUT = 1800  # refuse queued generation after 30 min


class ModelSlot:
    """The single loaded pipeline + its locks (load and generation)."""

    def __init__(self) -> None:
        self.load_lock = threading.Lock()
        self.gen_lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.pipe = None
        self.kind: str | None = None          # llm | vlm | image
        self.path: str | None = None
        self.device: str | None = None
        self.devices: list[str] | None = None  # segmented (image only)
        self.geometry: dict | None = None      # static geometry when segmented
        self.tokenizer = None
        self.loading = bool(False)
        self.error: str | None = None
        self.loaded_at: float | None = None

    def describe(self) -> dict:
        return {
            "loading": self.loading, "kind": self.kind, "path": self.path,
            "name": Path(self.path).name if self.path else None,
            "device": self.device, "devices": self.devices,
            "geometry": self.geometry, "error": self.error,
            "loaded": self.pipe is not None,
        }


SLOT = ModelSlot()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def scan_models(models_dir: str) -> list[dict]:
    """Inventory <models_dir>/<kind>/<model>/<variant>/ directories."""
    root = Path(models_dir)
    found: list[dict] = []
    if not root.is_dir():
        return found
    for kind in KIND_DIRS:
        kdir = root / kind
        if not kdir.is_dir():
            continue
        for mdir in sorted(kdir.iterdir()):
            if not mdir.is_dir():
                continue
            variants = []
            for vdir in sorted(mdir.iterdir()):
                if vdir.is_dir() and (any(vdir.rglob("openvino_*.xml"))
                                      or (vdir / "model_index.json").exists()):
                    try:
                        size_gb = sum(f.stat().st_size for f in vdir.rglob("*")
                                      if f.is_file()) / 1e9
                    except OSError:
                        size_gb = 0
                    variants.append({"name": vdir.name, "path": str(vdir),
                                     "size_gb": round(size_gb, 2)})
            if variants:
                found.append({"kind": kind, "name": mdir.name, "variants": variants})
    return found


def _b64_png(tensor_or_result) -> list[str]:
    """GenAI image result -> list of base64 PNG data URLs."""
    import numpy as np
    from PIL import Image

    import openvino as ov
    if isinstance(tensor_or_result, ov.Tensor):
        arr = np.asarray(tensor_or_result.data)
    elif hasattr(tensor_or_result, "images"):
        arr = np.stack([np.asarray(im.data if hasattr(im, "data") else im)
                        for im in tensor_or_result.images])
    elif hasattr(tensor_or_result, "data"):
        arr = np.asarray(tensor_or_result.data)
    else:
        raise RuntimeError(f"unsupported generate() result: {type(tensor_or_result)}")
    if arr.ndim == 3:
        arr = arr[None, ...]
    urls = []
    for i in range(arr.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(arr[i]).save(buf, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return urls


def _load_model(spec: dict) -> None:
    """Worker for /api/load; updates SLOT from a background thread."""
    try:
        SLOT.reset()
        SLOT.loading = True
        SLOT.path, SLOT.kind = spec["path"], spec["kind"]

        # registry gate before spending minutes on a compile
        fake_args = argparse.Namespace(
            model=spec["path"], device=spec["device"],
            devices=",".join(spec["devices"]) if spec.get("devices") else None,
            image=None, max_new_tokens=None, min_response_len=None)
        issues = registry_check(spec["kind"], spec["path"], spec["device"], fake_args)
        errors = [i.message for i in issues if i.level == "error"]
        if errors:
            raise SystemExit("registry: " + " | ".join(errors))

        if spec["kind"] == "image":
            import ovtool.imagegen as imagegen
            geo_args = argparse.Namespace(
                model=spec["path"], device=spec["device"],
                devices=",".join(spec["devices"]) if spec.get("devices") else None,
                num_images=int(spec.get("num_images", 1)),
                width=int(spec.get("width", 512)), height=int(spec.get("height", 512)),
                guidance_scale=float(spec.get("guidance_scale", 7.5)),
                opt=None, scheduler=None)
            pipe = imagegen._open_pipeline(ovgenai, geo_args, image_mode=False)
            SLOT.device = spec["device"]
            if spec.get("devices"):
                SLOT.devices = [d.upper() for d in spec["devices"]]
                SLOT.geometry = {k: geo_args.__dict__[k] for k in
                                 ("num_images", "width", "height", "guidance_scale")}
        else:
            pipe = ovgenai.LLMPipeline(spec["path"], device=spec["device"]) \
                if spec["kind"] == "llm" else \
                ovgenai.VLMPipeline(spec["path"], device=spec["device"])
            SLOT.device = spec["device"]

        SLOT.pipe = pipe
        try:
            SLOT.tokenizer = pipe.get_tokenizer()
        except Exception:
            SLOT.tokenizer = None
        SLOT.loaded_at = time.time()
    except BaseException as e:  # noqa: BLE001 - surface every failure to the UI
        SLOT.pipe = None
        SLOT.error = str(e)
    finally:
        SLOT.loading = False


class _ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class Handler(BaseHTTPRequestHandler):
    server_version = "ovtool-webui"
    protocol_version = "HTTP/1.1"

    models_dir: str = "./models"

    # ---------------- plumbing ---------------- #

    def log_message(self, fmt, *args):
        print(f"[webui] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else {}
            return body if isinstance(body, dict) else {}
        except json.JSONDecodeError as e:
            raise _ApiError(400, f"invalid JSON: {e}")

    # SSE (same chunked scheme as the serve command)
    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse(self, obj) -> bool:
        line = "data: [DONE]\n\n" if obj is None else \
            f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
        data = line.encode("utf-8")
        try:
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            return False

    def _sse_end(self) -> None:
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            pass
        self.close_connection = True

    # ---------------- routing ---------------- #

    def do_GET(self):
        try:
            path = self.path.split("?")[0]
            if path == "/":
                data = WEBUI_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/devices":
                # flatten non-serializable OpenVINO property values to strings
                rows = [{k: (v if isinstance(v, (str, int, float, bool, list)) else str(v))
                         for k, v in d.items()} for d in list_devices()]
                self._json(200, rows)
            elif path == "/api/models":
                self._json(200, scan_models(self.models_dir))
            elif path == "/api/status":
                self._json(200, SLOT.describe())
            else:
                raise _ApiError(404, f"unknown path: {path}")
        except _ApiError as e:
            self._json(e.status, {"error": e.message})
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True

    def do_POST(self):
        try:
            path = self.path.split("?")[0]
            body = self._read_json()
            if path == "/api/load":
                self._load(body)
            elif path == "/api/unload":
                self._unload()
            elif path == "/api/chat":
                self._chat(body)
            elif path == "/api/image":
                self._image(body)
            else:
                raise _ApiError(404, f"unknown path: {path}")
        except _ApiError as e:
            self._json(e.status, {"error": e.message})
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True

    # ---------------- endpoints ---------------- #

    def _load(self, body: dict) -> None:
        spec = {"path": body.get("path"), "kind": body.get("kind"),
                "device": (body.get("device") or "CPU").upper()}
        if spec["kind"] not in KIND_DIRS:
            raise _ApiError(400, "kind must be llm / vlm / image")
        if not spec["path"]:
            raise _ApiError(400, "path required")
        if not Path(spec["path"]).is_dir():
            raise _ApiError(400, f"not a model directory: {spec['path']}")
        if body.get("devices"):
            if spec["kind"] != "image":
                raise _ApiError(400, "segmented devices apply to image models only")
            spec["devices"] = [d.strip() for d in body["devices"].split(",") if d.strip()]
            if len(spec["devices"]) != 3:
                raise _ApiError(400, "devices expects TE,DENOISE,VAE")
            for k in ("num_images", "width", "height"):
                body[k] = int(body.get(k) or (1 if k == "num_images" else 512))
            body["guidance_scale"] = float(body.get("guidance_scale") or 7.5)
        for k in ("num_images", "width", "height", "guidance_scale"):
            if k in body:
                spec[k] = body[k]

        if not SLOT.load_lock.acquire(timeout=5):
            raise _ApiError(409, "another load is in progress")
        try:
            if SLOT.pipe is not None or SLOT.loading:
                _release()
            threading.Thread(target=_load_model, args=(spec,), daemon=True).start()
        finally:
            SLOT.load_lock.release()
        self._json(200, {"started": True})

    def _unload(self) -> None:
        if SLOT.loading:
            raise _ApiError(409, "load in progress")
        with SLOT.load_lock:
            _release()
        self._json(200, {"unloaded": True})

    def _require_chat(self):
        if SLOT.loading:
            raise _ApiError(409, "model is loading")
        if SLOT.pipe is None or SLOT.kind not in ("llm", "vlm"):
            raise _ApiError(400, "load an llm/vlm model first")
        if SLOT.error and SLOT.pipe is None:
            raise _ApiError(500, SLOT.error)

    def _chat(self, body: dict) -> None:
        self._require_chat()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise _ApiError(400, "messages required")
        images = None
        if body.get("images"):
            if SLOT.kind != "vlm":
                raise _ApiError(400, "image input requires a vlm model")
            import numpy as np
            from PIL import Image
            import openvino as ov
            images = []
            for b64 in body["images"]:
                try:
                    img = Image.open(io.BytesIO(base64.b64decode(b64.split(",")[-1])))
                    chw = np.asarray(img.convert("RGB")).transpose(2, 0, 1)
                    images.append(ov.Tensor(chw[None, ...]))
                except Exception as e:
                    raise _ApiError(400, f"bad image payload: {e}")

        if SLOT.tokenizer is not None:
            prompt = render_chat(SLOT.tokenizer, messages)
        else:  # pragma: no cover - tokenizers always expose get_tokenizer today
            prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        cfg = build_config({**body.get("params", {}),
                            "max_tokens": (body.get("params") or {}).get(
                                "max_tokens", DEFAULT_MAX_TOKENS)})

        stream = bool(body.get("stream", True))
        gen_kwargs = dict(generation_config=cfg)
        if images is not None:
            gen_kwargs["images"] = images
        # LLMPipeline: list input returns DecodedResults (perf metrics);
        # VLMPipeline takes the prompt string directly and returns metrics
        prompt_arg = prompt if SLOT.kind == "vlm" else [prompt]

        if not stream:
            with SLOT.gen_lock:
                result = SLOT.pipe.generate(prompt_arg, **gen_kwargs)
            text = result.texts[0] if hasattr(result, "texts") else str(result)
            usage = _usage(getattr(result, "perf_metrics", None))
            self._json(200, {"content": text, "usage": usage,
                             "finish_reason": _finish_reason(result, cfg)})
            return

        rid = "chat-" + uuid.uuid4().hex[:12]
        self._sse_start()
        alive = True
        with SLOT.gen_lock:

            def on_subword(subword):
                nonlocal alive
                if not isinstance(subword, str):
                    subword = str(subword)
                alive = self._sse({"id": rid, "delta": subword})
                return not alive  # stop generation on dead socket

            gen_kwargs["streamer"] = on_subword
            result = SLOT.pipe.generate(prompt_arg, **gen_kwargs)
        if alive:
            usage = _usage(getattr(result, "perf_metrics", None))
            self._sse({"id": rid, "usage": usage,
                       "finish_reason": _finish_reason(result, cfg)})
            self._sse(None)
        self._sse_end()

    def _image(self, body: dict) -> None:
        if SLOT.loading:
            raise _ApiError(409, "model is loading")
        if SLOT.pipe is None or SLOT.kind != "image":
            raise _ApiError(400, "load an image model first")
        prompt = body.get("prompt")
        if not prompt:
            raise _ApiError(400, "prompt required")
        g = SLOT.geometry or {}
        kwargs = dict(
            width=int(body.get("width", g.get("width", 512))),
            height=int(body.get("height", g.get("height", 512))),
            num_inference_steps=int(body.get("steps", 20)),
            guidance_scale=float(body.get("guidance_scale",
                                          g.get("guidance_scale", 7.5))),
            num_images_per_prompt=int(body.get("num_images", g.get("num_images", 1))))
        if SLOT.geometry and any(
                kwargs[k] != SLOT.geometry[gk] for k, gk in
                [("num_images_per_prompt", "num_images"), ("width", "width"),
                 ("height", "height"), ("guidance_scale", "guidance_scale")]):
            raise _ApiError(400, "segmented models are compiled for static geometry "
                                 f"{SLOT.geometry}; reload to change it")
        if body.get("negative_prompt"):
            kwargs["negative_prompt"] = body["negative_prompt"]
        if body.get("seed") is not None:
            kwargs["rng_seed"] = int(body["seed"])
        t0 = time.time()
        with SLOT.gen_lock:
            result = SLOT.pipe.generate(prompt, **kwargs)
        urls = _b64_png(result)
        out_dir = Path(body.get("out_dir", "./generated-webui"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        saved = []
        for i, u in enumerate(urls):
            p = out_dir / f"webui_{stamp}_{i}.png"
            p.write_bytes(base64.b64decode(u.split(",", 1)[1]))
            saved.append(str(p))
        self._json(200, {"images": urls, "saved": saved,
                         "elapsed_s": round(time.time() - t0, 1)})


def _release() -> None:
    """Drop the current pipeline so GPU/NPU memory is returned."""
    SLOT.reset()


def run_webui(args: argparse.Namespace) -> None:
    Handler.models_dir = args.models_dir
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[ovtool] webui at {url}  (models dir: {args.models_dir})")
    print("[ovtool] press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ovtool] webui stopped")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("webui", help="Browser UI for chat & image generation",
                       description="Start a local web UI: pick a converted model, "
                                   "load it on a device, chat (streaming) and generate "
                                   "images. One model is loaded at a time.\n"
                                   "Example:\n  ovtool webui --port 7860 --models-dir ./models",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    p.add_argument("--models-dir", default="./models",
                   help="Directory with <kind>/<model>/<variant>/ layout (default ./models)")
    p.set_defaults(func=run_webui)
