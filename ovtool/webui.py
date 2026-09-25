"""Browser UI for ovtool: chat & image generation on locally converted models.

`ovtool webui` starts an HTTP server that serves a single-page app (webui.html)
plus a small JSON API:

  GET  /                     the web UI
  GET  /api/devices          available OpenVINO devices
  GET  /api/models           models found under --models-dir
  GET  /api/status           currently loaded model / load progress
  POST /api/load             load a model (kind llm|vlm|image) on a device
  POST /api/unload           release the loaded pipeline
  POST /api/chat             chat completion (SSE stream or JSON), optionally
                             grounded in a knowledge base ("rag" options)
  POST /api/image            text-to-image generation (base64 PNG results)

  GET  /api/kb               knowledge bases with their documents
  POST /api/kb               create one (name, embedding model, chunking)
  DELETE /api/kb             delete one
  DELETE /api/kb/doc         remove a document
  POST /api/kb/ingest        queue uploads / pasted text / a local path
  GET  /api/kb/ingest        ingestion progress
  POST /api/kb/ingest/cancel drop queued documents, stop the current one
  POST /api/kb/search        retrieval preview with every candidate's scores
  GET  /api/retrieval        resident embedding / reranking models
  POST /api/retrieval/unload release them

One chat/image pipeline is loaded at a time (loading happens in a background
thread; the UI polls /api/status). Chat requests render the model's chat
template per request (stateless), reusing the serve implementation. Image
models may be loaded in segmented mode ("TE,DENOISE,VAE") which fixes static
geometry, mirroring `ovtool image --devices`.

Retrieval-augmented chat (see rag.py) keeps its embedding and reranking
pipelines resident beside the chat model, loaded on first use: the last user
turn is embedded, matched against the knowledge base, reranked, and the top
passages are spliced into that turn with numbered citations; the streamed
reply is preceded by the passages it was given.
"""
from __future__ import annotations

import argparse
import base64
import binascii
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

from . import rag
from .devices import list_devices
from .registry import check as registry_check, extra_model_roots, find_local_models
from .server import (DEFAULT_MAX_TOKENS, _content_to_text, _finish_reason, _set, _usage,
                     build_config, render_chat)

WEBUI_HTML = Path(__file__).parent / "webui.html"
KIND_DIRS = ("llm", "vlm", "image")        # loadable into the chat/image slot
CATALOG_KINDS = KIND_DIRS + ("embed", "rerank")  # + retrieval models (Knowledge tab)
GENERATION_LOCK_TIMEOUT = 1800  # refuse queued generation after 30 min
MAX_BODY_BYTES = 256 << 20  # document uploads arrive base64-encoded

# extra model roots added at runtime through the UI; when the
# persist_roots setting is on they are saved to SETTINGS_PATH and
# reloaded at startup. $OVTOOL_MODELS_PATH roots always apply on top.
SESSION_ROOTS: list[str] = []

SETTINGS_PATH = Path.home() / ".ovtool" / "webui_settings.json"
DEFAULT_SETTINGS = {"theme": "dark", "lang": "en", "persist": False}


def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_settings(data: dict) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[webui] warning: could not save settings: {e}")


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

# in-flight remote download (one at a time), driven by /api/download
DOWNLOAD: dict = {"active": False, "repo": None, "dest": None, "subfolder": None,
                  "endpoint": None, "files_done": 0, "files_total": 0,
                  "bytes_done": 0, "bytes_total": 0, "current": None,
                  "error": None, "done_at": None}
DOWNLOAD_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _resolve_eq(a: str, b: str) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return a == b


def all_roots(models_dir: str) -> list[str]:
    """Default dir + $OVTOOL_MODELS_PATH roots + session roots (deduped by
    resolved path, so the models dir cannot join twice via an absolute path)."""
    roots: list[str] = []
    for r in [models_dir] + SESSION_ROOTS + extra_model_roots():
        try:
            key = str(Path(r).resolve())
        except OSError:
            key = r
        if key not in roots:
            roots.append(key)
    # keep the configured default dir spelling first
    roots[0] = models_dir
    return roots


def scan_models(models_dir: str) -> list[dict]:
    """Inventory model directories under models_dir plus every extra root
    (session UI roots, $OVTOOL_MODELS_PATH), using recursive discovery."""
    seen: set[str] = set()
    groups: list[dict] = []
    for root in all_roots(models_dir):
        for g in find_local_models([root]):
            if g["kind"] not in CATALOG_KINDS:
                continue  # e.g. tts models: no web UI pipeline yet (ovtool tts)
            variants = [v for v in g["variants"] if v["path"] not in seen]
            seen.update(v["path"] for v in variants)
            if not variants:
                continue
            name = g["name"]
            if any(other["kind"] == g["kind"] and other["name"] == name
                   for other in groups):
                name = f"{name} ({Path(root).name})"
            groups.append({"kind": g["kind"], "name": name,
                           "source": root, "variants": variants})
    return groups


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
        SLOT.device = spec["device"]
        if spec.get("devices"):
            # surface the segmented placement immediately: NPU compilation
            # can take minutes, during which the UI must not claim the
            # whole pipeline runs on the device dropdown's value
            SLOT.devices = [d.strip().upper() for d in spec["devices"]]
            SLOT.geometry = {k: spec[k] for k in
                             ("num_images", "width", "height", "guidance_scale")
                             if k in spec}

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
            if not spec.get("devices"):
                SLOT.device = spec["device"]
            else:
                SLOT.devices = [d.upper() for d in spec["devices"]]
                SLOT.geometry = {k: geo_args.__dict__[k] for k in
                                 ("num_images", "width", "height", "guidance_scale")}
        else:
            # NPU static prompt budget matches `ovtool chat` (16384). VLM's
            # CLI default is 1024, which four RAG passages overflow.
            from .llm import open_llm_pipeline
            from .vlm import open_vlm_pipeline
            pipe = open_llm_pipeline(spec["path"], spec["device"]) \
                if spec["kind"] == "llm" else \
                open_vlm_pipeline(spec["path"], spec["device"],
                                  npu_prompt_len=16384)
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
    settings: dict = dict(DEFAULT_SETTINGS)
    # knowledge bases + resident retrieval models (set by run_webui)
    kb: rag.KBStore
    retriever: rag.Retriever
    ingestor: rag.Ingestor

    def _persist_now(self) -> None:
        """Write the settings file when persistence is on; drop it otherwise.

        The file carries the whole settings panel: theme, language, extra
        model roots and the saved UI state (selections, generation params).
        """
        if not self.settings.get("persist"):
            try:
                SETTINGS_PATH.unlink(missing_ok=True)
            except OSError:
                pass
            return
        data = {k: v for k, v in self.settings.items() if k != "persist_roots"}
        data["persist"] = True
        data["roots"] = list(SESSION_ROOTS)
        save_settings(data)

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
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self.close_connection = True  # the unread body cannot be skipped
            raise _ApiError(413, f"request body over {MAX_BODY_BYTES >> 20} MB")
        try:
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

    def _internal_error(self, e: Exception) -> None:
        """Answer an unexpected failure with a JSON 500 instead of dropping
        the socket (streaming routes report errors in-band themselves)."""
        self.log_error("%s %s failed: %r", self.command, self.path, e)
        self.close_connection = True
        try:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})
        except OSError:
            pass

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
            elif path == "/api/roots":
                self._json(200, {"default": self.models_dir,
                                 "env": extra_model_roots(),
                                 "session": list(SESSION_ROOTS)})
            elif path == "/api/settings":
                self._json(200, {k: v for k, v in self.settings.items()
                                 if k not in ("roots", "persist_roots")})
            elif path == "/api/download":
                self._json(200, DOWNLOAD)
            elif path == "/api/kb":
                self._json(200, self.kb.list())
            elif path == "/api/kb/ingest":
                self._json(200, self.ingestor.status())
            elif path == "/api/retrieval":
                self._json(200, self.retriever.describe())
            else:
                raise _ApiError(404, f"unknown path: {path}")
        except _ApiError as e:
            self._json(e.status, {"error": e.message})
        except rag.KBError as e:
            self._json(e.status, {"error": str(e)})
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True
        except Exception as e:  # noqa: BLE001
            self._internal_error(e)

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
            elif path == "/api/roots":
                self._add_root(body)
            elif path == "/api/settings":
                self._update_settings(body)
            elif path == "/api/download":
                self._start_download(body)
            elif path == "/api/kb":
                self._kb_create(body)
            elif path == "/api/kb/ingest":
                self._kb_ingest(body)
            elif path == "/api/kb/ingest/cancel":
                self._json(200, {"dropped": self.ingestor.cancel()})
            elif path == "/api/kb/search":
                self._kb_search(body)
            elif path == "/api/retrieval/unload":
                self.retriever.unload()
                self._json(200, {"unloaded": True})
            else:
                raise _ApiError(404, f"unknown path: {path}")
        except _ApiError as e:
            self._json(e.status, {"error": e.message})
        except rag.KBError as e:
            self._json(e.status, {"error": str(e)})
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True
        except Exception as e:  # noqa: BLE001
            self._internal_error(e)

    def do_DELETE(self):
        try:
            path = self.path.split("?")[0]
            if path == "/api/roots":
                self._remove_root(self._read_json())
            elif path == "/api/kb":
                self.kb.delete(str(self._read_json().get("id") or ""))
                self._json(200, {"deleted": True})
            elif path == "/api/kb/doc":
                body = self._read_json()
                self.kb.delete_document(str(body.get("kb") or ""), str(body.get("doc") or ""))
                self._json(200, {"deleted": True})
            else:
                raise _ApiError(404, f"unknown path: {self.path}")
        except _ApiError as e:
            self._json(e.status, {"error": e.message})
        except rag.KBError as e:
            self._json(e.status, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._internal_error(e)

    # ---------------- settings ---------------- #

    def _update_settings(self, body: dict) -> None:
        if "theme" in body and body["theme"] not in ("dark", "light"):
            raise _ApiError(400, "theme must be dark | light")
        if "lang" in body and body["lang"] not in ("en", "zh"):
            raise _ApiError(400, "lang must be en | zh")
        if "persist" in body and not isinstance(body["persist"], bool):
            raise _ApiError(400, "persist must be a boolean")
        if "ui" in body:
            if not isinstance(body["ui"], dict) or len(json.dumps(body["ui"])) > 8192:
                raise _ApiError(400, "ui must be a small JSON object")
            self.settings.setdefault("ui", {}).update(body["ui"])
        for key in ("theme", "lang", "persist"):
            if key in body:
                self.settings[key] = body[key]
        self._persist_now()
        self._json(200, {k: v for k, v in self.settings.items()
                         if k not in ("roots", "ui")})

    # ---------------- remote download ---------------- #

    def _start_download(self, body: dict) -> None:
        from .download import ENDPOINTS, _default_dest
        repo = (body.get("repo") or "").strip()
        dest = (body.get("dest") or "").strip() or _default_dest(repo, body.get("subfolder"))
        subfolder = (body.get("subfolder") or "").strip() or None
        endpoint = (body.get("endpoint") or "huggingface.co").strip()
        proxy = (body.get("proxy") or "").strip() or None
        if not repo or "/" not in repo:
            raise _ApiError(400, "repo must be a Hugging Face id like Qwen/Qwen3-0.6B")
        if endpoint not in ENDPOINTS:
            raise _ApiError(400, f"endpoint must be one of {', '.join(ENDPOINTS)}")
        if Path(dest).exists() and any(Path(dest).iterdir()):
            raise _ApiError(400, f"destination not empty: {dest}")
        with DOWNLOAD_LOCK:
            if DOWNLOAD["active"]:
                raise _ApiError(409, "a download is already running")
            DOWNLOAD.update({"active": True, "repo": repo, "dest": dest,
                             "subfolder": subfolder, "endpoint": endpoint,
                             "files_done": 0, "files_total": 0, "bytes_done": 0,
                             "bytes_total": 0, "current": None, "error": None,
                             "done_at": None})

        def worker():
            from .download import download_repo
            try:
                def on_progress(p):
                    DOWNLOAD.update(p)
                download_repo(repo, dest, endpoint=endpoint, subfolder=subfolder,
                              proxy=proxy, on_progress=on_progress)
                DOWNLOAD["done_at"] = time.time()
                # make the downloaded models immediately usable
                from .registry import find_local_models
                if find_local_models([dest]):
                    if not any(_resolve_eq(dest, r) for r in all_roots(self.models_dir)):
                        SESSION_ROOTS.append(dest)
                        self._persist_now()
                else:
                    print(f"[webui] download finished but no OpenVINO model "
                          f"detected under {dest}")
            except Exception as e:  # noqa: BLE001 - surfaced to the UI
                DOWNLOAD["error"] = str(e)
            finally:
                DOWNLOAD["active"] = False

        threading.Thread(target=worker, daemon=True).start()
        self._json(200, {"started": True, "dest": dest})

    # ---------------- model roots ---------------- #

    def _add_root(self, body: dict) -> None:
        path = (body.get("path") or "").strip().strip('"')
        if not path:
            raise _ApiError(400, "path required")
        root = Path(path)
        if not root.is_dir():
            raise _ApiError(400, f"not a directory: {path}")
        try:
            same_as_default = root.resolve() == Path(self.models_dir).resolve()
        except OSError:
            same_as_default = False
        if same_as_default or any(
                _resolve_eq(path, r) for r in all_roots(self.models_dir)):
            raise _ApiError(400, "root already active")
        found = find_local_models([path])
        if not found:
            raise _ApiError(400, f"no OpenVINO models found under {path} "
                                 "(any depth)")
        if path not in SESSION_ROOTS:
            SESSION_ROOTS.append(path)
            self._persist_now()
        count = sum(len(g["variants"]) for g in found)
        self._json(200, {"added": path, "models": len(found),
                         "variants": count})

    def _remove_root(self, body: dict) -> None:
        path = (body.get("path") or "").strip().strip('"')
        if path in extra_model_roots():
            raise _ApiError(400, "this root comes from $OVTOOL_MODELS_PATH; "
                                 "unset the environment variable instead")
        if path not in SESSION_ROOTS:
            raise _ApiError(404, "unknown session root")
        SESSION_ROOTS.remove(path)
        self._persist_now()
        self._json(200, {"removed": path})

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
        rag_opts = self._rag_options(body.get("rag"))
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

        cfg = build_config({**body.get("params", {}),
                            "max_tokens": (body.get("params") or {}).get(
                                "max_tokens", DEFAULT_MAX_TOKENS)})
        think = body.get("think", True) is not False
        if images is None:
            # _prompt() already rendered the template (and enable_thinking);
            # GenAI wrapping that string again would hide the variable.
            _set(cfg, "apply_chat_template", False)

        stream = bool(body.get("stream", True))
        gen_kwargs = dict(generation_config=cfg)
        if images is not None:
            gen_kwargs["images"] = images

        if not stream:
            rag_info = None
            if rag_opts:
                messages, rag_info = self._retrieve(messages, rag_opts)
            try:
                with SLOT.gen_lock:
                    result = SLOT.pipe.generate(
                        self._prompt(messages, think, with_images=bool(images)),
                        **gen_kwargs)
            except Exception as e:  # noqa: BLE001 - e.g. prompt over the NPU budget
                raise _ApiError(500, f"generation failed: {e}")
            text = result.texts[0] if hasattr(result, "texts") else str(result)
            usage = _usage(getattr(result, "perf_metrics", None))
            self._json(200, {"content": text, "usage": usage,
                             "finish_reason": _finish_reason(result, cfg),
                             "rag": rag_info})
            return

        rid = "chat-" + uuid.uuid4().hex[:12]
        self._sse_start()
        alive = True
        try:
            if rag_opts:
                # retrieval progress precedes the reply (a first query may
                # load the embedder / reranker), then the passages used
                alive = self._sse({"id": rid, "stage": "retrieve"})
                messages, rag_info = self._retrieve(
                    messages, rag_opts,
                    on_stage=lambda s, d=None: self._sse({"id": rid, "stage": s,
                                                          "detail": d}))
                alive = self._sse({"id": rid, "rag": rag_info})
            if alive:
                prompt_arg = self._prompt(messages, think, with_images=bool(images))
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
        except Exception as e:  # noqa: BLE001 - headers are out; report in-band
            self.log_error("chat failed: %r", e)
            if self._sse({"id": rid, "error": str(e)}):
                self._sse(None)
        self._sse_end()

    @staticmethod
    def _prompt(messages: list, think: bool = True, *, with_images: bool = False):
        extra = None if think else {"enable_thinking": False}
        if with_images:
            # VLMPipeline places image tokens while it applies the template;
            # ChatHistory carries enable_thinking as extra_context.
            history = [{"role": m.get("role", "user"),
                        "content": _content_to_text(m.get("content"))}
                       for m in messages]
            chat = ovgenai.ChatHistory(history)
            if extra:
                chat.set_extra_context(extra)
            return chat
        if SLOT.tokenizer is not None:
            prompt = render_chat(SLOT.tokenizer, messages, extra)
        else:  # pragma: no cover - tokenizers always expose get_tokenizer today
            prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        # LLMPipeline: list input returns DecodedResults (perf metrics);
        # VLMPipeline takes the prompt string directly and returns metrics
        return prompt if SLOT.kind == "vlm" else [prompt]

    # ---------------- knowledge bases (RAG) ---------------- #

    def _rag_options(self, raw) -> dict | None:
        """Validated retrieval options of a chat/search request (None = off)."""
        if not isinstance(raw, dict) or not raw.get("kb"):
            return None
        try:
            opts = {"kb": str(raw["kb"]),
                    "top_k": int(raw.get("top_k") or 20),
                    "top_n": int(raw.get("top_n") or 4),
                    "min_score": float(raw.get("min_score") or 0.0),
                    "reranker": str(raw["reranker"]) if raw.get("reranker") else None,
                    "embed_device": str(raw.get("embed_device") or "CPU").strip().upper(),
                    "rerank_device": str(raw.get("rerank_device") or "CPU").strip().upper()}
        except (TypeError, ValueError):
            raise _ApiError(400, "invalid retrieval options")
        if not 1 <= opts["top_n"] <= 20:
            raise _ApiError(400, "top_n (passages) must be 1-20")
        if not opts["top_n"] <= opts["top_k"] <= 200:
            raise _ApiError(400, "top_k (candidates) must be between top_n and 200")
        self.kb.get(opts["kb"])  # unknown KB -> 404 before any work
        return opts

    def _retrieve(self, messages: list, opts: dict, on_stage=None) -> tuple[list, dict]:
        """Ground the last user turn in the knowledge base: returns the
        messages with retrieved passages spliced into that turn (unchanged
        when nothing clears min_score) and the retrieval summary for the UI."""
        last = next((i for i in range(len(messages) - 1, -1, -1)
                     if isinstance(messages[i], dict) and messages[i].get("role") == "user"),
                    None)
        if last is None:
            raise _ApiError(400, "retrieval needs a user message")
        question = _content_to_text(messages[last].get("content")).strip()
        res = rag.search(self.kb, self.retriever, opts["kb"], rag.retrieval_query(question),
                         top_k=opts["top_k"], top_n=opts["top_n"],
                         min_score=opts["min_score"], reranker=opts["reranker"],
                         embed_device=opts["embed_device"],
                         rerank_device=opts["rerank_device"], on_stage=on_stage)
        if res["hits"]:
            messages = list(messages)
            messages[last] = {**messages[last],
                              "content": rag.augment(question, res["hits"], res["kb"]["name"])}
        # the candidate list is for the Knowledge tab's retrieval preview
        return messages, {k: v for k, v in res.items() if k != "candidates"}

    def _kb_create(self, body: dict) -> None:
        try:
            size = int(body.get("chunk_size") or rag.DEFAULT_CHUNK_SIZE)
            overlap = int(body.get("chunk_overlap") if body.get("chunk_overlap") is not None
                          else rag.DEFAULT_CHUNK_OVERLAP)
        except (TypeError, ValueError):
            raise _ApiError(400, "chunk_size / chunk_overlap must be integers")
        self._json(200, self.kb.create(str(body.get("name") or ""),
                                       str(body.get("embedder") or ""), size, overlap))

    def _kb_ingest(self, body: dict) -> None:
        items: list[dict] = []
        for f in body.get("files") or []:
            if not isinstance(f, dict):
                raise _ApiError(400, "files must be {name, data | text} objects")
            name = str(f.get("name") or "").strip()
            if not name:
                raise _ApiError(400, "every file needs a name")
            if "text" in f:  # pasted text
                items.append({"name": name, "text": str(f["text"]), "source": "paste"})
                continue
            try:
                data = base64.b64decode(f.get("data") or "")
            except (binascii.Error, ValueError) as e:
                raise _ApiError(400, f"bad base64 data for {name}: {e}")
            if len(data) > rag.MAX_FILE_BYTES:
                raise _ApiError(413, f"{name} is larger than {rag.MAX_FILE_BYTES >> 20} MB")
            items.append({"name": name, "data": data, "source": "upload"})
        if body.get("path"):
            items += rag.collect_files(str(body["path"]).strip().strip('"'))
        if not items:
            raise _ApiError(400, "nothing to ingest: send files, text or a path")
        device = str(body.get("embed_device") or "CPU").strip().upper()
        self._json(200, {"queued": self.ingestor.submit(str(body.get("kb") or ""),
                                                         items, device)})

    def _kb_search(self, body: dict) -> None:
        opts = self._rag_options(body)
        if opts is None:
            raise _ApiError(400, "kb required")
        query = str(body.get("query") or "").strip()
        if not query:
            raise _ApiError(400, "query required")
        self._json(200, rag.search(self.kb, self.retriever, opts["kb"], query,
                                   top_k=opts["top_k"], top_n=opts["top_n"],
                                   min_score=opts["min_score"], reranker=opts["reranker"],
                                   embed_device=opts["embed_device"],
                                   rerank_device=opts["rerank_device"]))

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
    stored = load_settings()
    Handler.settings = {**DEFAULT_SETTINGS,
                        **{k: v for k, v in stored.items() if k != "roots"}}
    # migrate the old per-path toggle
    if "persist_roots" in stored:
        Handler.settings["persist"] = bool(stored.get("persist_roots"))
    if Handler.settings.get("persist"):
        for r in stored.get("roots", []):
            if Path(r).is_dir() and r not in SESSION_ROOTS:
                SESSION_ROOTS.append(r)
    Handler.kb = rag.KBStore(args.kb_dir)
    Handler.retriever = rag.Retriever()
    Handler.ingestor = rag.Ingestor(Handler.kb, Handler.retriever)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[ovtool] webui at {url}  (models dir: {args.models_dir})")
    print(f"[ovtool] knowledge bases: {Path(args.kb_dir).resolve()}")
    print("[ovtool] press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ovtool] webui stopped")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("webui", help="Browser UI for chat, knowledge-base RAG & image generation",
                       description="Start a local web UI: pick a converted model, "
                                   "load it on a device, chat (streaming) and generate "
                                   "images. One chat/image model is loaded at a time; "
                                   "knowledge bases (embedding + reranking models) "
                                   "ground chat replies in your documents.\n"
                                   "Example:\n  ovtool webui --port 7860 --models-dir ./models",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    p.add_argument("--models-dir", default="./models",
                   help="Directory with <kind>/<model>/<variant>/ layout (default ./models)")
    p.add_argument("--kb-dir", default=str(rag.DEFAULT_KB_DIR),
                   help=f"Knowledge base storage (default {rag.DEFAULT_KB_DIR})")
    p.set_defaults(func=run_webui)
