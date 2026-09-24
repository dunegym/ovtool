"""Text embedding (vectorization) and reranking via openvino_genai pipelines.

- TextEmbeddingPipeline: embed_query / embed_documents with pooling and L2
  normalization; optional query/embed instruction prefixes (e5 / bge style)
- TextRerankPipeline: cross-encoder relevance scores (sigmoid) ranking
  candidate documents against a query

Pooling is not auto-detected by openvino-genai: `ovtool embed` reads the
sentence-transformers `1_Pooling/config.json` copied over by `convert embed`
and applies it, with `--pooling` as an explicit override.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import openvino_tokenizers  # noqa: F401  (registers custom-op extension)

POOLING_CONFIG = Path("1_Pooling") / "config.json"

# official Qwen3-Embedding / Qwen3-Reranker instruction prompts (model cards)
QWEN3_QUERY_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
QWEN3_RERANK_SYSTEM = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n')
QWEN3_RERANK_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def read_pooling(model_dir: str) -> str | None:
    """'cls' | 'mean' | 'last_token' | None from a sentence-transformers
    1_Pooling config."""
    try:
        cfg = json.loads((Path(model_dir) / POOLING_CONFIG).read_text(encoding="utf-8"))
        if cfg.get("pooling_mode_lasttoken") or cfg.get("pooling_mode_last_token"):
            return "last_token"  # decoder-based embedders (Qwen3-Embedding)
        if cfg.get("pooling_mode_cls_token") or cfg.get("pooling_mode_cls"):
            return "cls"
        if cfg.get("pooling_mode_mean_tokens") or cfg.get("pooling_mode_mean"):
            return "mean"
    except (OSError, ValueError):
        pass
    return None


def is_qwen3(model_dir: str) -> bool:
    """Qwen3-family export: model_type 'qwen3' in config.json (embedding and
    reranker models export as Qwen3ForCausalLM)."""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        return str(cfg.get("model_type", "")).lower() == "qwen3"
    except (OSError, ValueError):
        return False


def qwen3_rerank_wrap(query: str, texts: list[str], instruction: str | None):
    """Official Qwen3-Reranker prompt: openvino-genai feeds query+document to
    the model verbatim, so the instruction template must be applied by the
    caller (scores are near-random without it)."""
    q = (f"{QWEN3_RERANK_SYSTEM}<Instruct>: {instruction or QWEN3_QUERY_INSTRUCT}\n"
         f"<Query>: {query}\n")
    return q, [f"<Document>: {t}{QWEN3_RERANK_SUFFIX}" for t in texts]


def _pooling_enum(ovgenai, name: str):
    return {"cls": ovgenai.TextEmbeddingPipeline.PoolingType.CLS,
            "mean": ovgenai.TextEmbeddingPipeline.PoolingType.MEAN,
            "last_token": ovgenai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN}[name]


def _embed_config(ovgenai, args: argparse.Namespace, auto_pool: str | None):
    cfg = ovgenai.TextEmbeddingPipeline.Config()
    pooling = args.pooling or auto_pool
    if pooling:
        cfg.pooling_type = _pooling_enum(ovgenai, pooling)
    cfg.normalize = args.normalize
    if args.query_instruction:
        cfg.query_instruction = args.query_instruction
    if args.embed_instruction:
        cfg.embed_instruction = args.embed_instruction
    if args.max_length:
        cfg.max_length = args.max_length
    if args.batch_size:
        cfg.batch_size = args.batch_size
    return cfg


def _preview(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in v[:4]) + ", ...]"


def run_embed(args: argparse.Namespace) -> None:
    import openvino_genai as ovgenai

    from .devices import resolve_device
    from .llm import compile_options

    device = resolve_device(args.device)
    auto_pool = read_pooling(args.model)
    qwen3 = is_qwen3(args.model)
    if qwen3 and auto_pool is None:
        auto_pool = "last_token"  # decoder-based Qwen3-Embedding pooling
    if qwen3 and not args.query_instruction:
        # official retrieval instruction; measurably sharpens query ranking
        args.query_instruction = QWEN3_QUERY_INSTRUCT
    pooling = args.pooling or auto_pool or "mean"
    print(f"[ovtool] embedding model {args.model} on {device} "
          f"(pooling: {pooling}{' (from 1_Pooling config)' if auto_pool and not args.pooling else ''}) ...")
    cfg = _embed_config(ovgenai, args, auto_pool)
    pipe = ovgenai.TextEmbeddingPipeline(
        args.model, device, cfg, **compile_options(args))

    if args.query and args.texts:
        # retrieval mode: rank documents by cosine similarity to the query
        q = np.asarray(pipe.embed_query(args.query), dtype=np.float32)
        docs = [np.asarray(v, dtype=np.float32)
                for v in pipe.embed_documents(args.texts)]
        qn = np.linalg.norm(q) or 1.0
        scored = [(float(np.dot(q, d) / (qn * (np.linalg.norm(d) or 1.0))), i)
                  for i, d in enumerate(docs)]
        for rank, (score, i) in enumerate(sorted(scored, reverse=True), 1):
            print(f"{rank}. {score:+.4f}  {args.texts[i]}")
        return

    texts = args.texts or [args.query]
    vectors = [np.asarray(v, dtype=np.float32)
               for v in pipe.embed_documents(texts)]
    if args.json:
        payload = {"model": str(args.model), "dim": int(vectors[0].size),
                   "pooling": pooling, "normalized": args.normalize,
                   "embeddings": [{"text": t, "vector": v.tolist()}
                                  for t, v in zip(texts, vectors)]}
        out = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(out, encoding="utf-8")
            print(f"[ovtool] {len(texts)} vector(s), dim {vectors[0].size} -> {args.out}")
        else:
            print(out)
        return
    for t, v in zip(texts, vectors):
        print(f"dim={v.size}  norm={np.linalg.norm(v):.4f}  {_preview(v)}  {t!r}")


def run_rerank(args: argparse.Namespace) -> None:
    import openvino_genai as ovgenai

    from .devices import resolve_device
    from .llm import compile_options

    device = resolve_device(args.device)
    texts = args.texts
    if is_qwen3(args.model):
        # GenAI feeds query+document to the model verbatim; Qwen3-Reranker
        # needs its official instruction template or scores are meaningless
        query, texts = qwen3_rerank_wrap(args.query, texts, args.instruction)
        print("[ovtool] Qwen3 reranker: official instruction template applied "
              "(--instruction to customize)")
    else:
        query = args.query
    print(f"[ovtool] reranker {args.model} on {device} "
          f"({len(texts)} candidate(s)) ...")
    cfg = ovgenai.TextRerankPipeline.Config()
    cfg.top_n = args.top_n if args.top_n else len(texts)
    pipe = ovgenai.TextRerankPipeline(
        args.model, device, cfg, **compile_options(args))
    results = pipe.rerank(query, texts)
    for rank, (idx, score) in enumerate(results, 1):
        print(f"{rank}. {score:+.4f}  {args.texts[idx]!r}")
    if args.json:
        print(json.dumps([{"rank": r, "index": i, "score": s, "text": args.texts[i]}
                          for r, (i, s) in enumerate(results, 1)],
                         ensure_ascii=False, indent=2))


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", required=True,
                   help="Model directory (OpenVINO IR from 'convert embed/rerank')")
    p.add_argument("-d", "--device", default="CPU",
                   help="Inference device (default CPU)")
    p.add_argument("--opt", action="append", metavar="KEY=VALUE",
                   help="Runtime option, e.g. --opt perf_mode=THROUGHPUT (repeatable)")


def add_parsers(sub: argparse._SubParsersAction) -> None:
    p1 = sub.add_parser("embed", help="Text embeddings (vectorization)",
                        description="Embed texts (or a query) with a converted embedding model.\n"
                                    "Examples:\n"
                                    "  ovtool embed -m ./bge-small \"OpenVINO is a toolkit\" \"A cat video\"\n"
                                    "  ovtool embed -m ./bge-small --query \"what is OpenVINO?\" doc1.txt-contents doc2.txt-contents  # cosine ranking\n"
                                    "  ovtool embed -m ./bge-small \"text\" --json --out vec.json",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    _common_args(p1)
    p1.add_argument("texts", nargs="*", default=[], help="Texts to embed (documents)")
    p1.add_argument("--query", default=None,
                    help="Embed one query (embed_query path); when combined with texts, "
                         "documents are ranked by cosine similarity to it")
    p1.add_argument("--pooling", choices=["cls", "mean", "last_token"], default=None,
                    help="Pooling strategy (default: from the model's 1_Pooling config "
                         "when present, else mean; Qwen3-Embedding uses last_token)")
    p1.add_argument("--no-normalize", dest="normalize", action="store_false", default=True,
                    help="Disable L2 normalization of the embeddings")
    p1.add_argument("--query-instruction", default=None,
                    help="Instruction prefixed to queries (e.g. bge: "
                         "'Represent this sentence for searching relevant passages: ')")
    p1.add_argument("--embed-instruction", default=None,
                    help="Instruction prefixed to embedded documents (e.g. e5: 'passage: ')")
    p1.add_argument("--max-length", type=int, default=None, help="Max token length")
    p1.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p1.add_argument("--json", action="store_true", help="Emit full vectors as JSON")
    p1.add_argument("--out", default=None, help="Write --json output to this file")
    p1.set_defaults(func=run_embed)

    p2 = sub.add_parser("rerank", help="Rerank documents against a query (cross-encoder)",
                        description="Score and rank candidate documents for a query with a "
                                    "converted reranker (sigmoid relevance scores).\n"
                                    "Example:\n"
                                    "  ovtool rerank -m ./bge-reranker \"what is OpenVINO?\" \\\n"
                                    "      \"OpenVINO is an inference toolkit\" \"A cat video\"",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    _common_args(p2)
    p2.add_argument("query", help="The query text")
    p2.add_argument("texts", nargs="+", help="Candidate documents")
    p2.add_argument("--top-n", type=int, default=None,
                    help="Return only the top N documents (default: all)")
    p2.add_argument("--instruction", default=None,
                    help="Qwen3-Reranker <Instruct> text (default: web-search retrieval instruction)")
    p2.add_argument("--json", action="store_true", help="Additionally emit results as JSON")
    p2.set_defaults(func=run_rerank)
