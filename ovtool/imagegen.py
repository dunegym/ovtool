"""Image generation (text-to-image / image-to-image) via openvino_genai pipelines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# registers the openvino_tokenizers custom-op extension with the runtime
import openvino_tokenizers  # noqa: F401


def _save_images(result, out_dir: Path, prefix: str) -> list[str]:
    """Save generated images. openvino_genai (2026.x) returns a single ov.Tensor
    shaped (N, H, W, C) uint8; older versions returned a result object with .data/.images."""
    import openvino as ov
    from PIL import Image

    if isinstance(result, ov.Tensor):
        arr = np.asarray(result.data)
    elif hasattr(result, "images"):
        arr = np.stack([np.asarray(im.data if hasattr(im, "data") else im) for im in result.images])
    elif hasattr(result, "data"):
        arr = np.asarray(result.data)
    else:
        raise SystemExit(f"Unsupported generate() result type: {type(result)}")
    if arr.ndim == 3:
        arr = arr[None, ...]
    saved = []
    for i in range(arr.shape[0]):
        path = out_dir / f"{prefix}_{i}.png"
        Image.fromarray(arr[i]).save(path)
        saved.append(str(path))
    return saved


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", required=True, help="Model directory (OpenVINO IR from 'convert image')")
    p.add_argument("-d", "--device", default="GPU", help="Inference device (default GPU;扩散模型推荐 GPU)")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=20, help="num_inference_steps (default 20)")
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--num-images", type=int, default=1, help="num_images_per_prompt")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--scheduler", default=None,
                   help="Override scheduler type, e.g. LCM, DPM_SOLIDER_MULTISTEP, EULER_ANCESTRAL")
    p.add_argument("--out-dir", default="./generated", help="Directory for generated PNGs")
    p.add_argument("--devices", default=None, metavar="TE,DENOISE,VAE",
                   help="Segmented execution: one device per pipeline component, e.g. "
                        "--devices NPU,NPU,GPU (text encoder + denoiser on NPU, VAE decode "
                        "on GPU). Fixes static shapes to --width/--height. VAE decode is "
                        "not NPU-capable; keep the third entry on GPU/CPU.")
    p.add_argument("--opt", action="append", metavar="KEY=VALUE", help="Runtime option KEY=VALUE (repeatable)")


def _open_pipeline(ovgenai, args, image_mode: bool):
    from .devices import resolve_device
    device = resolve_device(args.device)
    opts = {}
    for kv in args.opt or []:
        if "=" not in kv:
            raise SystemExit(f"--opt expects KEY=VALUE, got: {kv}")
        k, v = kv.split("=", 1)
        opts[k] = int(v) if v.isdigit() else v

    cls = ovgenai.Image2ImagePipeline if image_mode else ovgenai.Text2ImagePipeline

    if getattr(args, "devices", None):
        # Segmented execution: each pipeline component gets its own device.
        # Static shapes are mandatory here (NPU has no dynamic-shape
        # support), so the pipeline is reshaped to the requested geometry
        # before compiling.
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        if len(devices) != 3:
            raise SystemExit("--devices expects three entries: TEXT_ENCODER,DENOISER,VAE "
                             "(e.g. NPU,NPU,GPU)")
        devices = [resolve_device(d).upper() for d in devices]
        if "NPU" in devices:
            # NPU compilation is slow; cache compiled graphs next to the model
            opts.setdefault("CACHE_DIR", str(Path(args.model) / "cache"))
        print(f"[ovtool] segmented pipeline: text_encoder={devices[0]} "
              f"denoiser={devices[1]} vae={devices[2]} "
              f"@ {args.num_images}x{args.width}x{args.height}")
        pipe = cls(args.model)
        if not hasattr(pipe, "reshape") or not hasattr(pipe, "compile"):
            raise SystemExit("this openvino-genai release does not support "
                             "per-component compile(); update openvino-genai")
        pipe.reshape(int(args.num_images), int(args.height), int(args.width),
                     float(args.guidance_scale))
        pipe.compile(devices[0], devices[1], devices[2],
                     **({"config": opts} if opts else {}))
    else:
        kwargs = dict(device=device)
        if opts:
            kwargs["config"] = opts
        pipe = cls(args.model, **kwargs)
    if args.scheduler:
        sched = getattr(ovgenai.SchedulerType, args.scheduler.upper(), None)
        if sched is None:
            raise SystemExit(f"Unknown scheduler '{args.scheduler}'. "
                             f"Options: {[s.name for s in ovgenai.SchedulerType]}")
        pipe.set_scheduler(sched)
    return pipe


def run_text2image(args: argparse.Namespace) -> None:
    import openvino_genai as ovgenai
    pipe = _open_pipeline(ovgenai, args, image_mode=False)
    gen_kwargs = dict(width=args.width, height=args.height,
                      num_inference_steps=args.steps,
                      guidance_scale=args.guidance_scale,
                      num_images_per_prompt=args.num_images)
    if args.negative_prompt:
        gen_kwargs["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        gen_kwargs["rng_seed"] = args.seed
    print(f"[ovtool] generating {args.num_images} image(s) on "
          f"{args.device} ({args.steps} steps) ...")
    result = pipe.generate(args.prompt, **gen_kwargs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = _save_images(result, out_dir, "t2i")
    for s in saved:
        print("saved:", s)


def run_image2image(args: argparse.Namespace) -> None:
    import openvino_genai as ovgenai
    from PIL import Image
    pipe = _open_pipeline(ovgenai, args, image_mode=True)
    image = Image.open(args.image).convert("RGB")
    image_tensor = ov_genai_tensor(image)
    gen_kwargs = dict(num_inference_steps=args.steps,
                      guidance_scale=args.guidance_scale,
                      num_images_per_prompt=args.num_images)
    if args.negative_prompt:
        gen_kwargs["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        gen_kwargs["rng_seed"] = args.seed
    print(f"[ovtool] img2img on {args.device} ({args.steps} steps) ...")
    result = pipe.generate(args.prompt, image_tensor, **gen_kwargs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = _save_images(result, out_dir, "i2i")
    for s in saved:
        print("saved:", s)


def ov_genai_tensor(pil_image):
    """Image tensor for Image2ImagePipeline: uint8 NHWC with batch dim."""
    import openvino as ov
    return ov.Tensor(np.asarray(pil_image)[None, ...])


def add_parsers(sub: argparse._SubParsersAction) -> None:
    p1 = sub.add_parser("image", help="Text-to-image generation (SD/SDXL/Flux ...)",
                        description="Text-to-image with a converted diffusion model. "
                                    "GPU is the recommended device for diffusion models.")
    _common_args(p1)
    p1.add_argument("prompt", help="Positive prompt")
    p1.set_defaults(func=run_text2image)

    p2 = sub.add_parser("image2image", help="Image-to-image generation",
                        description="Image-to-image generation with a converted diffusion model.")
    _common_args(p2)
    p2.add_argument("-i", "--image", required=True, help="Input image path")
    p2.add_argument("prompt", help="Positive prompt")
    p2.set_defaults(func=run_image2image)
