"""LLM text generation via openvino_genai.LLMPipeline."""
from __future__ import annotations

import argparse

import openvino_genai as ovgenai
import openvino_tokenizers  # noqa: F401  (registers custom-op extension)


def make_streamer(stream: bool):
    """Return a streamer callback; openvino_genai (2024.5+) calls it with each
    decoded subword string."""
    if not stream:
        return None

    def _put(subword):
        if isinstance(subword, str):
            print(subword, end="", flush=True)
        else:  # very old API passed raw token ids
            print(str(subword), end="", flush=True)
        return None  # continue generation

    return _put


def build_generation_config(args: argparse.Namespace) -> ovgenai.GenerationConfig:
    cfg = ovgenai.GenerationConfig()
    cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
        cfg.do_sample = args.temperature > 0
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.repetition_penalty is not None:
        cfg.repetition_penalty = args.repetition_penalty
    if args.rng_seed is not None:
        cfg.rng_seed = args.rng_seed
    if args.stop_tokens:
        cfg.stop_strings = args.stop_tokens
    return cfg


def compile_options(args: argparse.Namespace) -> dict:
    """Extra device runtime options (--opt KEY=VALUE, repeatable) mapped to
    OpenVINO property names accepted by GenAI pipeline constructors."""
    opts: dict = {}
    for kv in args.opt or []:
        if "=" not in kv:
            raise SystemExit(f"--opt expects KEY=VALUE, got: {kv}")
        k, v = kv.split("=", 1)
        key, val = k.strip(), v.strip()
        lk = key.lower()
        if lk in ("perf_mode", "performance_hint"):
            aliases = {"LOW_LATENCY": "LATENCY", "HIGH_THROUGHPUT": "THROUGHPUT"}
            val = aliases.get(val.upper(), val.upper())
            if val not in ("LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"):
                raise SystemExit(f"Invalid perf_mode {val!r}; "
                                 "use LATENCY / THROUGHPUT / CUMULATIVE_THROUGHPUT")
            opts["PERFORMANCE_HINT"] = val
        elif lk == "inference_num_threads":
            opts["INFERENCE_NUM_THREADS"] = int(val)
        elif lk == "num_streams":
            opts["NUM_STREAMS"] = val if val.lower() == "auto" else int(val)
        elif lk == "enable_hyper_threading":
            opts["ENABLE_HYPER_THREADING"] = val.lower() in ("1", "true", "yes")
        elif lk == "cache_dir":
            opts["CACHE_DIR"] = val
        else:
            opts[key] = int(val) if val.isdigit() else val  # pass through
    return opts


def open_llm_pipeline(model_dir: str, device: str, opts: dict | None = None,
                      npu_shape: tuple[int, int] | None = None) -> ovgenai.LLMPipeline:
    opts = dict(opts or {})
    # NPU executes LLMs with static shapes; the prompt/response budget must be
    # fixed at compile time via device properties
    if device.upper().startswith("NPU"):
        max_prompt, min_response = npu_shape or (16384, 256)
        opts.setdefault("MAX_PROMPT_LEN", int(max_prompt))
        opts.setdefault("MIN_RESPONSE_LEN", int(min_response))
        opts.setdefault("PERFORMANCE_HINT", "LATENCY")
    return ovgenai.LLMPipeline(model_dir, device=device, **({"config": opts} if opts else {}))


def _npu_shape_from_args(args: argparse.Namespace):
    if getattr(args, "max_prompt_len", None) or getattr(args, "min_response_len", None):
        return (args.max_prompt_len or 16384, args.min_response_len or 256)
    return None


def run_generate(args: argparse.Namespace) -> None:
    from .devices import resolve_device
    device = resolve_device(args.device)
    opts = compile_options(args)
    print(f"[ovtool] loading LLM {args.model} on {device} ...")
    pipe = open_llm_pipeline(args.model, device, opts, _npu_shape_from_args(args))
    cfg = build_generation_config(args)
    streamer = make_streamer(args.stream)
    # list input -> DecodedResults (so perf metrics are available); str input
    # would shortcut to a plain str
    result = pipe.generate([args.prompt], generation_config=cfg, streamer=streamer)
    if not args.stream:
        print(result.texts[0])
    if args.stats:
        stats = result.perf_metrics
        print(f"\n[stats] TTFT {stats.get_ttft().mean:.1f}ms | "
              f"TPOT {stats.get_tpot().mean:.2f}ms/tok | "
              f"throughput {stats.get_throughput().mean:.1f} tok/s | "
              f"total {stats.get_generate_duration().mean:.0f}ms")


def run_chat(args: argparse.Namespace) -> None:
    from .devices import resolve_device
    device = resolve_device(args.device)
    opts = compile_options(args)
    print(f"[ovtool] loading LLM {args.model} on {device} ...")
    pipe = open_llm_pipeline(args.model, device, opts, _npu_shape_from_args(args))
    pipe.start_chat()
    cfg = build_generation_config(args)
    print("Interactive chat. Commands: /exit quit, /reset clear history, /system <text> set system prompt")
    try:
        while True:
            try:
                user = input("\nYou > ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user == "/exit":
                break
            if user == "/reset":
                pipe.finish_chat()
                pipe.start_chat()
                print("[chat history cleared]")
                continue
            if user.startswith("/system "):
                pipe.set_chat_history([{"role": "system", "content": user[len("/system "):]}])
                print("[system prompt set]")
                continue
            print("Assistant > ", end="", flush=True)
            streamer = make_streamer(True)
            pipe.generate(user, generation_config=cfg, streamer=streamer)
    finally:
        pipe.finish_chat()


def add_parsers(sub: argparse._SubParsersAction) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-m", "--model", required=True, help="Model directory (OpenVINO IR from 'convert')")
    common.add_argument("-d", "--device", default="CPU",
                        help="Inference device: CPU / GPU / NPU / AUTO / HETERO:GPU,CPU ... (default CPU)")
    common.add_argument("--opt", action="append", metavar="KEY=VALUE",
                        help="Runtime option, e.g. --opt perf_mode=THROUGHPUT --opt inference_num_threads=8 (repeatable)")
    common.add_argument("--max-prompt-len", type=int, default=None,
                        help="NPU only: static prompt budget for compile (default 16384). "
                             "Requires a symmetric-INT4 model (convert with --sym).")
    common.add_argument("--min-response-len", type=int, default=None,
                        help="NPU only: static response budget for compile (default 256)")
    gen = argparse.ArgumentParser(add_help=False)
    gen.add_argument("--max-new-tokens", type=int, default=512)
    gen.add_argument("--temperature", type=float, default=None, help=">0 enables sampling")
    gen.add_argument("--top-p", type=float, default=None)
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("--repetition-penalty", type=float, default=None)
    gen.add_argument("--rng-seed", type=int, default=None)
    gen.add_argument("--stop-tokens", nargs="*", default=None, help="Stop strings")

    p1 = sub.add_parser("generate", parents=[common, gen],
                        help="One-shot LLM text generation",
                        description="Single-prompt generation with an OpenVINO LLM.")
    p1.add_argument("prompt", help="The prompt text")
    p1.add_argument("--no-stream", dest="stream", action="store_false", default=True)
    p1.add_argument("--stats", action="store_true", help="Print TTFT/TPOT/throughput")
    p1.set_defaults(func=run_generate)

    p2 = sub.add_parser("chat", parents=[common, gen],
                        help="Interactive multi-turn chat",
                        description="Interactive chat with history kept on the pipeline side.")
    p2.set_defaults(func=run_chat)
