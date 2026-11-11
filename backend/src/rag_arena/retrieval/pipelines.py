"""The five retrieval pipelines.

Every pipeline returns the top `k` chunk ids for a question plus per-stage timings. CPU-bound
steps run in worker threads so the API can run pipelines concurrently; the benchmark runs
them one at a time, so its latency numbers are not inflated by contention.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from rag_arena.llm.base import Usage
from rag_arena.retrieval.backend import SearchBackend
from rag_arena.retrieval.fusion import RRF_K, reciprocal_rank_fusion
from rag_arena.retrieval.rerank import Reranker
from rag_arena.retrieval.rewrite import Rewriter

PIPELINES: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rewrite")

PIPELINE_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
    "hybrid_rewrite": "Hybrid + rewrite",
    # Reranker variants benchmarked for retrieval only.
    "hybrid_rerank_minilm": "Hybrid + MiniLM rerank",
    "hybrid_rerank_jina": "Hybrid + Jina rerank",
}


@dataclass(frozen=True)
class PipelineConfig:
    k: int = 10  # results returned (metrics use up to 10, the generator gets the top 5)
    candidates: int = 50  # depth of each BM25 / dense list before fusion
    rerank_depth: int = 20  # hybrid results re-scored by the cross-encoder
    rrf_k: int = RRF_K


@dataclass
class Hit:
    chunk_id: str
    score: float


@dataclass
class RetrievalResult:
    pipeline: str
    chunk_set: str
    hits: list[Hit]
    latency_ms: float
    timings: dict[str, float] = field(default_factory=dict)
    rewrites: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)  # LLM tokens spent during retrieval
    llm_model: str | None = None

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "chunk_set": self.chunk_set,
            "hits": [[h.chunk_id, round(h.score, 6)] for h in self.hits],
            "latency_ms": round(self.latency_ms, 3),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "rewrites": self.rewrites,
            "usage": self.usage.to_dict(),
            "llm_model": self.llm_model,
        }


class PipelineError(RuntimeError):
    pass


class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    async def run[T](self, name: str, fn: Callable[[], T]) -> T:
        t0 = time.perf_counter()
        out = await asyncio.to_thread(fn)
        self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return out


async def _hybrid_lists(
    backend: SearchBackend, queries: list[str], cfg: PipelineConfig, timer: _Timer
) -> list[list[str]]:
    """BM25 and dense rankings for each query, in a fixed order: bm25(q1), dense(q1), ..."""
    vectors = await timer.run("embed", lambda: backend.embed_queries(queries))
    lists: list[list[str]] = []
    for q, vec in zip(queries, vectors, strict=True):
        bm = await timer.run("bm25", partial(backend.bm25_search, q, cfg.candidates))
        dn = await timer.run("dense", partial(backend.dense_search, vec, cfg.candidates))
        lists.append([i for i, _ in bm])
        lists.append([i for i, _ in dn])
    return lists


async def run_pipeline(
    name: str,
    question: str,
    backend: SearchBackend,
    *,
    cfg: PipelineConfig | None = None,
    reranker: Reranker | None = None,
    rewriter: Rewriter | None = None,
) -> RetrievalResult:
    cfg = cfg or PipelineConfig()
    timer = _Timer()
    rewrites: list[str] = []
    usage = Usage()
    llm_model: str | None = None
    # Wall time spent waiting on the rewrite call (rate limiting, retries, or a cache hit),
    # replaced below by the latency of the successful model call itself.
    rewrite_wall_ms = 0.0
    rewrite_model_ms = 0.0
    t0 = time.perf_counter()

    if name == "bm25":
        ranked = await timer.run("bm25", lambda: backend.bm25_search(question, cfg.k))
        hits = [Hit(i, s) for i, s in ranked]
    elif name == "dense":
        vec = (await timer.run("embed", lambda: backend.embed_queries([question])))[0]
        ranked = await timer.run("dense", lambda: backend.dense_search(vec, cfg.k))
        hits = [Hit(i, s) for i, s in ranked]
    elif name in ("hybrid", "hybrid_rerank"):
        lists = await _hybrid_lists(backend, [question], cfg, timer)
        depth = cfg.rerank_depth if name == "hybrid_rerank" else cfg.k
        fused = reciprocal_rank_fusion(lists, k=cfg.rrf_k, limit=max(depth, cfg.k))
        if name == "hybrid":
            hits = [Hit(i, s) for i, s in fused[: cfg.k]]
        else:
            if reranker is None:
                raise PipelineError("hybrid_rerank needs a reranker")
            candidates = [i for i, _ in fused[: cfg.rerank_depth]]
            texts = [backend.chunks[i].index_text for i in candidates]
            scores = await timer.run("rerank", lambda: reranker.score(question, texts))
            # Stable sort keeps the hybrid order among equal scores.
            order = sorted(range(len(candidates)), key=lambda j: -scores[j])
            hits = [Hit(candidates[j], scores[j]) for j in order[: cfg.k]]
    elif name == "hybrid_rewrite":
        if rewriter is None:
            raise PipelineError("hybrid_rewrite needs a rewriter")
        t_rw = time.perf_counter()
        rw = await rewriter.rewrite(question)
        rewrite_wall_ms = (time.perf_counter() - t_rw) * 1000
        rewrite_model_ms = rw.latency_ms
        timer.timings["rewrite"] = rw.latency_ms
        rewrites, usage, llm_model = rw.rewrites, rw.usage, rw.model
        lists = await _hybrid_lists(backend, [question, *rewrites], cfg, timer)
        fused = reciprocal_rank_fusion(lists, k=cfg.rrf_k, limit=cfg.k)
        hits = [Hit(i, s) for i, s in fused]
    else:
        raise PipelineError(f"unknown pipeline {name!r}; choose from {PIPELINES}")

    return RetrievalResult(
        pipeline=name,
        chunk_set=backend.set_id,
        hits=hits,
        latency_ms=(time.perf_counter() - t0) * 1000 - rewrite_wall_ms + rewrite_model_ms,
        timings=timer.timings,
        rewrites=rewrites,
        usage=usage,
        llm_model=llm_model,
    )
