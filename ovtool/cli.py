"""ovtool entry point: comprehensive OpenVINO GenAI command line tool."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    from . import __version__
    parser = argparse.ArgumentParser(
        prog="ovtool",
        description="OpenVINO GenAI command-line tool: LLM (incl. multimodal) inference, "
                    "image generation, device selection, model conversion & quantization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Typical workflow:\n"
               "  ovtool devices\n"
               "  ovtool convert llm Qwen/Qwen2.5-0.5B-Instruct -m ./qwen05-int4\n"
               "  ovtool chat -m ./qwen05-int4 -d GPU --opt perf_mode=LOW_LATENCY\n"
               "  ovtool image -m ./sd-turbo-int8 \"a corgi surfing a wave\" --steps 8 --seed 42\n")
    parser.add_argument("--version", action="version", version=f"ovtool {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # devices
    from .devices import print_devices
    from .registry import print_models
    p = sub.add_parser("models", help="List registry-known HF models with verified device/parameter combos")
    p.add_argument("query", nargs="?", default=None, help="Optional case-insensitive filter")
    p.set_defaults(func=lambda args: print_models(args.query))

    p = sub.add_parser("devices", help="List available inference devices")
    p.set_defaults(func=lambda args: print_devices())

    # convert
    from .convert import add_convert_parser
    add_convert_parser(sub)

    # llm
    from .llm import add_parsers as add_llm_parsers
    add_llm_parsers(sub)

    # vlm
    from .vlm import add_parser as add_vlm_parser
    add_vlm_parser(sub)

    # serve (OpenAI-compatible API)
    from .server import add_parser as add_serve_parser
    add_serve_parser(sub)

    # image
    from .imagegen import add_parsers as add_image_parsers
    add_image_parsers(sub)

    args = parser.parse_args(argv)
    hook = getattr(args, "hook", None)
    if hook:
        hook(args)

    # registry compatibility gate: reject known-bad (model x device x params) combos
    kind_by_cmd = {"generate": "llm", "chat": "llm", "serve": "llm", "vlm": "vlm",
                   "image": "image", "image2image": "image"}
    kind = kind_by_cmd.get(args.command)
    if kind and getattr(args, "model", None) and getattr(args, "device", None):
        from .registry import check, report
        if not report(check(kind, args.model, args.device, args)):
            print("Blocked by the compatibility registry. Run 'ovtool models' for verified combos.")
            sys.exit(2)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)


if __name__ == "__main__":
    main()
