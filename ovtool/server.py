"""OpenAI-compatible HTTP API server on top of openvino_genai.LLMPipeline.

Endpoints:
  GET  /health                  liveness probe
  GET  /v1/models               the single served model
  POST /v1/chat/completions     chat endpoint (stateless: the chat template is
                                rendered per request, no server-side history)
  POST /v1/completions          legacy text completions

Both generation endpoints support `stream: true` (SSE over chunked transfer
encoding) and the common sampling parameters (temperature / top_p / top_k /
max_tokens / stop / seed / n). Responses include token `usage` taken from
GenAI perf metrics.

Only stdlib + openvino-genai are used; a single pipeline is shared across
requests and serialized with a lock (generation is not thread-safe).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import openvino_genai as ovgenai
import openvino_tokenizers  # noqa: F401  (registers custom-op extension)

from .llm import _npu_shape_from_args, compile_options, open_llm_pipeline

DEFAULT_MAX_TOKENS = 512


# --------------------------------------------------------------------------- #
# request -> GenAI mapping
# --------------------------------------------------------------------------- #

def _set(cfg: ovgenai.GenerationConfig, attr: str, value) -> None:
    """Set a GenerationConfig field, ignoring params this GenAI lacks."""
    try:
        setattr(cfg, attr, value)
    except Exception:
        pass


def build_config(body: dict) -> ovgenai.GenerationConfig:
    cfg = ovgenai.GenerationConfig()
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") \
        or DEFAULT_MAX_TOKENS
    cfg.max_new_tokens = int(max_tokens)
    if body.get("temperature") is not None:
        cfg.temperature = float(body["temperature"])
        cfg.do_sample = float(body["temperature"]) > 0
    if body.get("top_p") is not None:
        _set(cfg, "top_p", float(body["top_p"]))
    if body.get("top_k") is not None:
        _set(cfg, "top_k", int(body["top_k"]))
    if body.get("repetition_penalty") is not None:
        _set(cfg, "repetition_penalty", float(body["repetition_penalty"]))
    if body.get("frequency_penalty"):
        _set(cfg, "frequency_penalty", float(body["frequency_penalty"]))
    if body.get("presence_penalty"):
        _set(cfg, "presence_penalty", float(body["presence_penalty"]))
    if body.get("seed") is not None:
        _set(cfg, "rng_seed", int(body["seed"]))
    stop = body.get("stop")
    if stop:
        cfg.stop_strings = [stop] if isinstance(stop, str) else [str(s) for s in stop]
    return cfg


def _content_to_text(content) -> str:
    """OpenAI content may be a plain string or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


def render_chat(tokenizer, messages: list[dict]) -> str:
    history = [{"role": m.get("role", "user"),
                "content": _content_to_text(m.get("content"))}
               for m in messages]
    return tokenizer.apply_chat_template(history, add_generation_prompt=True)


def _usage(perf_metrics) -> dict:
    try:
        pt = int(perf_metrics.get_num_input_tokens())
        ct = int(perf_metrics.get_num_generated_tokens())
    except Exception:
        pt = ct = 0
    return {"prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": pt + ct}


def _finish_reason(result, cfg: ovgenai.GenerationConfig) -> str:
    try:  # GenAI exposes per-request finish reasons on newer releases
        reasons = [str(r) for r in getattr(result, "meta", None) or []]
        if any("LENGTH" in r.upper() for r in reasons):
            return "length"
    except Exception:
        pass
    try:
        if int(result.perf_metrics.get_num_generated_tokens()) >= cfg.max_new_tokens:
            return "length"
    except Exception:
        pass
    return "stop"


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #

class _ApiError(Exception):
    def __init__(self, status: int, message: str,
                 err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.err_type = status, message, err_type


class Handler(BaseHTTPRequestHandler):
    server_version = "ovtool-serve"
    protocol_version = "HTTP/1.1"  # enables keep-alive + chunked streaming

    # injected by run_serve()
    pipeline: ovgenai.LLMPipeline
    tokenizer: ovgenai.Tokenizer
    gen_lock: threading.Lock
    model_name: str
    api_key: str | None
    created: int

    # ---------------- plumbing ---------------- #

    def log_message(self, fmt, *args):  # single concise line
        print(f"[serve] {self.address_string()} {fmt % args}", flush=True)

    def _auth(self) -> None:
        if self.api_key:
            header = self.headers.get("Authorization", "")
            if header != f"Bearer {self.api_key}":
                raise _ApiError(401, "Invalid or missing API key",
                                "authentication_error")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise _ApiError(400, "Request body required")
        if length > 10 * 1024 * 1024:
            raise _ApiError(413, "Request body too large")
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            raise _ApiError(400, f"Invalid JSON: {e}")
        if not isinstance(body, dict):
            raise _ApiError(400, "Request body must be a JSON object")
        return body

    def _send_json(self, status: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- SSE over chunked transfer encoding ---- #

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse_safe(self, obj: dict | None) -> bool:
        """Write one SSE event (None -> the terminal [DONE]); False on a dead
        socket so streamers can stop generation instead of looping on EPIPE."""
        line = "data: [DONE]\n\n" if obj is None else \
            f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
        data = line.encode("utf-8")
        try:
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError,
                OSError):
            return False

    def _sse_end(self) -> None:
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError,
                OSError):
            pass
        self.close_connection = True

    # ---------------- routing ---------------- #

    def do_GET(self):
        try:
            self._auth()
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", ""):
                self._send_json(200, {"status": "ok", "model": self.model_name})
            elif path == "/v1/models" or path == "/v1/models/" + self.model_name:
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": self.model_name, "object": "model",
                              "created": self.created, "owned_by": "ovtool"}]})
            else:
                raise _ApiError(404, f"Unknown endpoint: {path}")
        except _ApiError as e:
            self._send_json(e.status, {"error": {"message": e.message,
                                                 "type": e.err_type}})
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True

    def do_POST(self):
        try:
            self._auth()
            path = self.path.split("?")[0].rstrip("/")
            body = self._read_json()
            if path == "/v1/chat/completions":
                self._chat_completions(body)
            elif path == "/v1/completions":
                self._completions(body)
            else:
                raise _ApiError(404, f"Unknown endpoint: {path}")
        except _ApiError as e:
            self._send_json(e.status, {"error": {"message": e.message,
                                                 "type": e.err_type}})
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True

    # ---------------- endpoints ---------------- #

    def _validate_common(self, body: dict):
        if body.get("tools") or body.get("tool_choice"):
            raise _ApiError(400, "tools / function calling are not supported "
                                 "by this server")
        if body.get("logprobs"):
            raise _ApiError(400, "logprobs are not supported")
        try:
            n = int(body.get("n") or 1)
        except (TypeError, ValueError):
            raise _ApiError(400, "'n' must be an integer")
        if n < 1 or n > 8:
            raise _ApiError(400, "'n' must be between 1 and 8")
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {})
                             .get("include_usage"))
        return n, stream, include_usage, build_config(body)

    @staticmethod
    def _chunk_base(rid: str, obj_type: str, created: int) -> dict:
        return {"id": rid, "object": obj_type, "created": created,
                "model": Handler.model_name}

    def _chat_completions(self, body: dict) -> None:
        n, stream, include_usage, cfg = self._validate_common(body)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise _ApiError(400, "'messages' must be a non-empty array")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m:
                raise _ApiError(400, "each message needs a 'role'")
            if isinstance(m.get("content"), list) and \
                    any(p.get("type") != "text" for p in m["content"]
                        if isinstance(p, dict)):
                raise _ApiError(400, "only text content parts are supported "
                                     "by the LLM server (use 'ovtool vlm' "
                                     "for images)")
        prompt = render_chat(self.tokenizer, messages)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if not stream:
            texts: list[str] = []
            prompt_tokens = completion_tokens = 0
            finish = "stop"
            with self.gen_lock:
                for _ in range(n):
                    result = self.pipeline.generate([prompt], generation_config=cfg)
                    texts.append(result.texts[0])
                    u = _usage(result.perf_metrics)
                    prompt_tokens += u["prompt_tokens"]
                    completion_tokens += u["completion_tokens"]
                    finish = _finish_reason(result, cfg)
            self._send_json(200, {
                "id": rid, "object": "chat.completion", "created": created,
                "model": self.model_name,
                "choices": [{"index": i, "message": {"role": "assistant",
                                                     "content": t},
                             "finish_reason": finish}
                            for i, t in enumerate(texts)],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}})
            return

        # streaming: choices are generated one after another (single pipeline)
        self._sse_start()
        alive = True
        with self.gen_lock:
            for idx in range(n):
                alive = self._sse_safe({
                    **self._chunk_base(rid, "chat.completion.chunk", created),
                    "choices": [{"index": idx, "delta": {"role": "assistant"},
                                 "finish_reason": None}]})
                if not alive:
                    break

                def on_subword(subword, idx=idx):
                    if not isinstance(subword, str):
                        subword = str(subword)
                    ok = self._sse_safe({
                        **self._chunk_base(rid, "chat.completion.chunk", created),
                        "choices": [{"index": idx, "delta": {"content": subword},
                                     "finish_reason": None}]})
                    return not ok  # truthy return stops generation

                result = self.pipeline.generate([prompt],
                                                generation_config=cfg,
                                                streamer=on_subword)
                alive = self._sse_safe({
                    **self._chunk_base(rid, "chat.completion.chunk", created),
                    "choices": [{"index": idx, "delta": {},
                                 "finish_reason": _finish_reason(result, cfg)}]})
                if not alive:
                    break
        if alive:
            if include_usage:
                self._sse_safe({**self._chunk_base(rid, "chat.completion.chunk",
                                                   created),
                                "choices": [],
                                "usage": _usage(result.perf_metrics)})
            self._sse_safe(None)  # data: [DONE]
        self._sse_end()

    def _completions(self, body: dict) -> None:
        n, stream, include_usage, cfg = self._validate_common(body)
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            if any(not isinstance(p, str) for p in prompt):
                raise _ApiError(400, "token-based prompts are not supported; "
                                     "pass strings")
        elif isinstance(prompt, str):
            prompt = [prompt]
        else:
            raise _ApiError(400, "'prompt' must be a string or array of strings")
        if stream and (len(prompt) > 1 or n > 1):
            raise _ApiError(400, "streaming requires a single prompt and n=1")
        rid = "cmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if not stream:
            texts: list[str] = []
            prompt_tokens = completion_tokens = 0
            finish = "stop"
            with self.gen_lock:
                for _ in range(n):
                    result = self.pipeline.generate(prompt, generation_config=cfg)
                    texts.extend(result.texts)
                    u = _usage(result.perf_metrics)
                    prompt_tokens += u["prompt_tokens"]
                    completion_tokens += u["completion_tokens"]
                    finish = _finish_reason(result, cfg)
            self._send_json(200, {
                "id": rid, "object": "text_completion", "created": created,
                "model": self.model_name,
                "choices": [{"index": i, "text": t, "logprobs": None,
                             "finish_reason": finish}
                            for i, t in enumerate(texts)],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}})
            return

        def text_chunk(choice: dict) -> bool:
            return self._sse_safe({**self._chunk_base(rid, "text_completion.chunk",
                                                      created),
                                   "choices": [choice]})

        self._sse_start()
        alive = text_chunk({"index": 0, "text": "", "logprobs": None,
                            "finish_reason": None})
        with self.gen_lock:
            if alive:

                def on_subword(subword):
                    if not isinstance(subword, str):
                        subword = str(subword)
                    ok = text_chunk({"index": 0, "text": subword,
                                     "logprobs": None, "finish_reason": None})
                    return not ok  # truthy return stops generation

                result = self.pipeline.generate(prompt, generation_config=cfg,
                                                streamer=on_subword)
                alive = text_chunk({"index": 0, "text": "", "logprobs": None,
                                    "finish_reason": _finish_reason(result, cfg)})
        if alive:
            if include_usage:
                self._sse_safe({**self._chunk_base(rid, "text_completion.chunk",
                                                   created),
                                "choices": [], "usage": _usage(result.perf_metrics)})
            self._sse_safe(None)  # data: [DONE]
        self._sse_end()


def run_serve(args: argparse.Namespace) -> None:
    from .devices import resolve_device
    device = resolve_device(args.device)
    opts = compile_options(args)
    print(f"[ovtool] loading LLM {args.model} on {device} ...")
    pipe = open_llm_pipeline(args.model, device, opts, _npu_shape_from_args(args))

    Handler.pipeline = pipe
    Handler.tokenizer = pipe.get_tokenizer()
    Handler.gen_lock = threading.Lock()
    Handler.model_name = Path(args.model.rstrip("/\\")).name
    Handler.api_key = args.api_key
    Handler.created = int(time.time())

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[ovtool] serving {Handler.model_name!r} (OpenAI-compatible API)")
    print(f"[ovtool]   POST {url}/v1/chat/completions")
    print(f"[ovtool]   POST {url}/v1/completions")
    print(f"[ovtool]   GET  {url}/v1/models")
    print("[ovtool] press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ovtool] server stopped")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("serve", help="Serve an LLM behind an OpenAI-compatible API",
                       description="Start an OpenAI-compatible HTTP server "
                                   "(/v1/chat/completions, /v1/completions, /v1/models) "
                                   "backed by an OpenVINO LLM.\n"
                                   "Example:\n"
                                   "  ovtool serve -m ./qwen3-06b-int4 -d GPU --port 8000\n"
                                   "  curl http://127.0.0.1:8000/v1/chat/completions "
                                   "-d '{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True,
                   help="Model directory (OpenVINO IR from 'convert')")
    p.add_argument("-d", "--device", default="CPU",
                   help="Inference device (default CPU)")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    p.add_argument("--api-key", default=None,
                   help="If set, require 'Authorization: Bearer <key>' on every request")
    p.add_argument("--opt", action="append", metavar="KEY=VALUE",
                   help="Runtime option, e.g. --opt perf_mode=THROUGHPUT (repeatable)")
    p.add_argument("--max-prompt-len", type=int, default=None,
                   help="NPU only: static prompt budget for compile (default 16384)")
    p.add_argument("--min-response-len", type=int, default=None,
                   help="NPU only: static response budget for compile (default 256)")
    p.set_defaults(func=run_serve)
