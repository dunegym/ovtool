"""Download models from the Hugging Face Hub to a local directory.

Two endpoints are supported: huggingface.co (default) and hf-mirror.com
(a mirror reachable without proxies in some networks). Downloads go through
the huggingface_hub cache (so interrupted runs resume per file) and are
copied into the destination preserving the repo layout; an optional
`subfolder` restricts the download to one model inside a multi-model repo
(e.g. a single quantization variant) and is stripped from the destination
layout.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ENDPOINTS = ("huggingface.co", "hf-mirror.com")


def _endpoint_url(endpoint: str) -> str:
    if endpoint.startswith("https://"):
        return endpoint
    if endpoint not in ENDPOINTS:
        raise SystemExit(f"unknown endpoint {endpoint!r}; "
                         f"use {' or '.join(ENDPOINTS)} (or a full https:// URL)")
    return f"https://{endpoint}"


def _apply_endpoint(endpoint: str) -> None:
    """Point huggingface_hub at the chosen endpoint. The library reads
    constants.ENDPOINT when building URLs, but only at import time for some
    helpers — set both the env var and the module constant."""
    url = _endpoint_url(endpoint)
    os.environ["HF_ENDPOINT"] = url
    import huggingface_hub.constants as hub_constants
    hub_constants.ENDPOINT = url


def _entry_size(entry) -> int | None:
    """Real transfer size of a repo tree entry (LFS blobs report the pointer
    file size; the actual payload lives in entry.lfs)."""
    lfs = getattr(entry, "lfs", None)
    if lfs is not None and getattr(lfs, "size", None):
        return int(lfs.size)
    size = getattr(entry, "size", None)
    return int(size) if size is not None else None


def list_repo_files(repo: str, endpoint: str, subfolder: str | None = None,
                    revision: str = "main") -> list[tuple[str, int]]:
    """[(repo_path, transfer_bytes)] for every file, optionally under a
    subfolder only."""
    from huggingface_hub import HfApi
    api = HfApi(endpoint=_endpoint_url(endpoint))
    prefix = subfolder.strip("/").rstrip("/") if subfolder else None
    files: list[tuple[str, int]] = []
    for entry in api.list_repo_tree(repo, revision=revision, recursive=True):
        size = _entry_size(entry)
        if size is None:  # directories
            continue
        path = entry.path
        if prefix and not (path == prefix or path.startswith(prefix + "/")):
            continue
        files.append((path, size))
    return files


def download_repo(repo: str, dest: str, endpoint: str = "huggingface.co",
                  subfolder: str | None = None, revision: str = "main",
                  proxy: str | None = None,
                  on_progress=None) -> dict:
    """Download `repo` (or its `subfolder`) into `dest`.

    on_progress(dict) is called after each file with
    {files_done, files_total, bytes_done, bytes_total, current}.
    Returns a summary {files, bytes, dest}.
    """
    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["HTTP_PROXY"] = proxy
        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    _apply_endpoint(endpoint)

    files = list_repo_files(repo, endpoint, subfolder, revision)
    if not files:
        raise RuntimeError(
            f"no files found in {repo!r}"
            + (f" under {subfolder!r}" if subfolder else ""))

    prefix = subfolder.strip("/").rstrip("/") + "/" if subfolder else ""
    dest_path = Path(dest)
    bytes_total = sum(size for _, size in files)
    bytes_done = 0
    from huggingface_hub import hf_hub_download
    for done, (path, size) in enumerate(files, 1):
        cached = hf_hub_download(repo_id=repo, filename=path, revision=revision)
        rel = path[len(prefix):] if prefix else path
        target = dest_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
        bytes_done += size
        if on_progress:
            on_progress({"files_done": done, "files_total": len(files),
                         "bytes_done": bytes_done, "bytes_total": bytes_total,
                         "current": rel})
    return {"files": len(files), "bytes": bytes_done, "dest": str(dest_path)}


def _default_dest(repo: str, subfolder: str | None) -> str:
    if subfolder:
        name = subfolder.strip("/").rstrip("/").split("/")[-1]
        return f"./downloads/{name}"
    return f"./downloads/{repo.split('/')[-1]}"


def run_download(args: argparse.Namespace) -> None:
    dest = args.output or _default_dest(args.repo, args.subfolder)
    print(f"[ovtool] downloading {args.repo!r}"
          + (f" ({args.subfolder})" if args.subfolder else "")
          + f" from {args.endpoint} -> {dest}")
    if args.proxy:
        print(f"[ovtool] via proxy {args.proxy}")

    def report(p: dict) -> None:
        pct = 100 * p["bytes_done"] / max(1, p["bytes_total"])
        print(f"[{p['files_done']:>3}/{p['files_total']}] {pct:5.1f}% "
              f"{p['current']}", flush=True)

    summary = download_repo(args.repo, dest, endpoint=args.endpoint,
                            subfolder=args.subfolder, revision=args.revision,
                            proxy=args.proxy, on_progress=report)
    print(f"[ovtool] done: {summary['files']} files, "
          f"{summary['bytes'] / 1e9:.2f} GB -> {summary['dest']}")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("download", help="Download a model from the Hugging Face Hub",
                       description="Download a HF repo (or one subfolder of it) to a "
                                   "local directory. Works with pre-converted OpenVINO "
                                   "repos and raw transformers/diffusers repos alike.\n"
                                   "Examples:\n"
                                   "  ovtool download Qwen/Qwen3-0.6B -o ./qwen3\n"
                                   "  ovtool download dunegym/openvino-models "
                                   "--subfolder llm/Qwen3-0.6B/int4-sym-g128 -o ./qwen3-sym\n"
                                   "  ovtool download ... --endpoint hf-mirror.com",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", help="Hugging Face repo id, e.g. Qwen/Qwen3-0.6B")
    p.add_argument("-o", "--output", default=None,
                   help="Destination directory (default ./downloads/<name>)")
    p.add_argument("--subfolder", default=None,
                   help="Restrict the download to this repo subfolder (its "
                        "contents become the destination root)")
    p.add_argument("--endpoint", default="huggingface.co", choices=list(ENDPOINTS),
                   help="Hub endpoint (default huggingface.co; hf-mirror.com "
                        "routes around blocked networks)")
    p.add_argument("--revision", default="main", help="Repo revision (default main)")
    p.add_argument("--proxy", default=None,
                   help="HTTP(S) proxy for this download, e.g. http://host:3128")
    p.set_defaults(func=run_download)
